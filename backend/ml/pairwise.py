"""Pairwise ranking evaluation — the primary research instrument.

Instead of "did we predict the winner?", scores every driver pair per race:
did the model order A ahead of B correctly? ~190 pairs/race x 71 races gives
two orders of magnitude more statistical resolution than race-level metrics,
and decomposes into buckets that name WHERE the model struggles.

Conventions:
- CLEAN pairs: neither driver DNF'd. All headline numbers use clean pairs only.
- CONTAMINATED pairs (either DNF'd) are reported separately — attrition luck
  must not be mixed into skill-ordering accuracy.
- Buckets overlap (a pair can be both "teammates" and "adjacent"); each bucket
  is reported independently with race-clustered bootstrap CIs.

Run from backend/:  python -m ml.pairwise
Writes data/models/pairwise.md and pairwise.json (the frozen reference for
future experiment comparisons via compare()).
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd

import config
from ml.dataset import DATASET_PARQUET
from ml.evaluate import RaceResult, _blend, _load_or_run
from ml.features import FEATURE_COLS
from ml.train import BLEND_W, PARAMS

PAIRWISE_MD_PATH = config.MODELS_DIR / "pairwise.md"
PAIRWISE_JSON_PATH = config.MODELS_DIR / "pairwise.json"

BUCKETS = [
    "all_clean", "teammates", "similar_cars", "adjacent_1_2", "mid_3_5",
    "distant_6plus", "top4_internal", "midfield_internal", "backmarker_internal",
    "cross_tier", "comeback_involved", "contaminated_dnf",
]


def build_pair_table(races: list[RaceResult], blend_w: float = BLEND_W) -> pd.DataFrame:
    """One row per driver pair per race, with correctness and bucket flags."""
    ds = pd.read_parquet(DATASET_PARQUET)
    meta = ds.set_index(["season", "round", "driver_code"])[["team", "team_season_points"]]

    rows = []
    for rc in races:
        s = _blend(rc["raw"], rc["grid"], blend_w)
        pred_rank = s.argsort().argsort() + 1
        actual = rc["actual"]
        codes = rc["driver_codes"]
        n = len(codes)

        def team_of(i: int) -> tuple[str | None, float]:
            try:
                m = meta.loc[(rc["season"], rc["round"], codes[i])]
                return str(m["team"]), float(m["team_season_points"])
            except KeyError:
                return None, np.nan

        teams = [team_of(i) for i in range(n)]
        # rank teams by prior season points for the "similar machinery" bucket
        pts = sorted({t[1] for t in teams if not np.isnan(t[1])}, reverse=True)
        team_rank = {p: r for r, p in enumerate(pts)}
        gain = rc["grid"] - actual  # positions gained (grid is NaN-filled upstream)

        for i, j in combinations(range(n), 2):
            correct = (pred_rank[i] - pred_rank[j]) * (actual[i] - actual[j]) > 0
            dnf_pair = bool(rc["dnf"][i] or rc["dnf"][j])
            lo, hi = sorted((actual[i], actual[j]))
            margin = hi - lo
            ti, tj = teams[i], teams[j]
            same_team = ti[0] is not None and ti[0] == tj[0]
            similar = (
                not same_team
                and not (np.isnan(ti[1]) or np.isnan(tj[1]))
                and abs(team_rank.get(ti[1], 99) - team_rank.get(tj[1], -99)) <= 1
            )
            rows.append({
                "season": rc["season"], "round": rc["round"],
                "race": f"{rc['season']} R{rc['round']}",
                "pair": "–".join(sorted((codes[i], codes[j]))),
                "correct": bool(correct), "dnf_pair": dnf_pair,
                "teammates": same_team, "similar_cars": bool(similar),
                "margin": int(margin),
                "both_top4": bool(hi <= 4),
                "both_midfield": bool(lo >= 5 and hi <= 14),
                "both_back": bool(lo >= 15),
                "comeback": bool(max(gain[i], gain[j]) >= 5),
            })
    return pd.DataFrame(rows)


def _bucket_mask(d: pd.DataFrame, bucket: str) -> pd.Series:
    clean = ~d["dnf_pair"]
    return {
        "all_clean": clean,
        "teammates": clean & d["teammates"],
        "similar_cars": clean & d["similar_cars"],
        "adjacent_1_2": clean & d["margin"].le(2),
        "mid_3_5": clean & d["margin"].between(3, 5),
        "distant_6plus": clean & d["margin"].ge(6),
        "top4_internal": clean & d["both_top4"],
        "midfield_internal": clean & d["both_midfield"],
        "backmarker_internal": clean & d["both_back"],
        "cross_tier": clean & ~d["both_top4"] & ~d["both_midfield"] & ~d["both_back"],
        "comeback_involved": clean & d["comeback"],
        "contaminated_dnf": d["dnf_pair"],
    }[bucket]


def bucket_summary(d: pd.DataFrame, n_boot: int = 500, seed: int = 42) -> pd.DataFrame:
    """Accuracy per bucket with race-clustered bootstrap CIs (pairs within a
    race are correlated, so resampling races — not pairs — is the honest CI)."""
    rng = np.random.default_rng(seed)
    race_ids = d["race"].unique()
    by_race = {b: d[_bucket_mask(d, b)].groupby("race")["correct"].agg(["sum", "size"])
               for b in BUCKETS}
    rows = []
    for b in BUCKETS:
        g = by_race[b]
        n, acc = int(g["size"].sum()), float(g["sum"].sum() / max(g["size"].sum(), 1))
        boots = []
        for _ in range(n_boot):
            sample = rng.choice(race_ids, size=len(race_ids), replace=True)
            gg = g.reindex(sample).dropna()
            tot = gg["size"].sum()
            if tot:
                boots.append(gg["sum"].sum() / tot)
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
        rows.append({"bucket": b, "n_pairs": n, "accuracy": acc,
                     "ci_lo": float(lo), "ci_hi": float(hi)})
    return pd.DataFrame(rows)


def most_reversed_pairs(d: pd.DataFrame, min_meetings: int = 8) -> pd.DataFrame:
    """Recurring driver pairs the model most often orders wrongly (clean only)."""
    clean = d[~d["dnf_pair"]]
    g = clean.groupby("pair")["correct"].agg(meetings="size", correct="sum")
    g = g[g["meetings"] >= min_meetings]
    g["reversal_rate"] = 1 - g["correct"] / g["meetings"]
    return (g.sort_values("reversal_rate", ascending=False)
             .head(15).reset_index())


def compare(table_a: pd.DataFrame, table_b: pd.DataFrame,
            n_boot: int = 1000, seed: int = 42) -> dict:
    """Race-clustered bootstrap of the clean pairwise-accuracy delta (B − A).
    The standard significance test for future experiments."""
    rng = np.random.default_rng(seed)
    ga = table_a[~table_a["dnf_pair"]].groupby("race")["correct"].agg(["sum", "size"])
    gb = table_b[~table_b["dnf_pair"]].groupby("race")["correct"].agg(["sum", "size"])
    races = ga.index.intersection(gb.index).to_numpy()
    deltas = []
    for _ in range(n_boot):
        sample = rng.choice(races, size=len(races), replace=True)
        sa, sb = ga.loc[sample], gb.loc[sample]
        deltas.append(sb["sum"].sum() / sb["size"].sum() - sa["sum"].sum() / sa["size"].sum())
    point = float(gb["sum"].sum() / gb["size"].sum() - ga["sum"].sum() / ga["size"].sum())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "significant": bool(lo > 0 or hi < 0)}


def render_markdown(summary: pd.DataFrame, reversed_pairs: pd.DataFrame,
                    n_races: int) -> str:
    lines = [
        "# Pairwise ranking report",
        "",
        f"{n_races} backtest races, blend w={BLEND_W}. Clean = neither driver DNF'd.",
        "CIs are race-clustered bootstrap (95%).",
        "",
        "| bucket | pairs | accuracy | 95% CI |",
        "|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(f"| {r.bucket} | {r.n_pairs} | {r.accuracy:.1%} | "
                     f"{r.ci_lo:.1%}–{r.ci_hi:.1%} |")
    lines += ["", "## Most-reversed recurring pairs (min 8 clean meetings)", "",
              "| pair | meetings | reversal rate |", "|---|---|---|"]
    for _, r in reversed_pairs.iterrows():
        lines.append(f"| {r.pair} | {r.meetings} | {r.reversal_rate:.0%} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    config.ensure_dirs()
    df = pd.read_parquet(DATASET_PARQUET)
    races, cache_hit = _load_or_run(df, FEATURE_COLS, PARAMS, use_cache=True)
    print(f"walk-forward: {len(races)} races ({'cache hit' if cache_hit else 'refit'})")

    table = build_pair_table(races)
    summary = bucket_summary(table)
    rev = most_reversed_pairs(table)
    md = render_markdown(summary, rev, len(races))
    print("\n" + md)

    PAIRWISE_MD_PATH.write_text(md)
    PAIRWISE_JSON_PATH.write_text(json.dumps({
        "n_races": len(races), "blend_w": BLEND_W,
        "buckets": summary.to_dict(orient="records"),
        "most_reversed": rev.to_dict(orient="records"),
    }, indent=2))
    print(f"saved {PAIRWISE_MD_PATH} and {PAIRWISE_JSON_PATH}")


if __name__ == "__main__":
    main()
