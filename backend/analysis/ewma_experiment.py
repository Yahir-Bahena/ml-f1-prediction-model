"""Experiment E3: momentum / EWMA + trend features for the F1 race-winner model.

Adds 7 driver/team momentum features (already wired into ml.features.driver_features
and rebuilt into dataset.parquet) and scores them with the PROJECT-STANDARD
framework (ml.evaluate.run_walk_forward / evaluate / to_dict / render_comparison).
Nothing here mutates production; adoption (editing FEATURE_COLS + retrain) is done
separately, gated on the adoption rule printed at the end.

Run from backend/ with the project venv:
    ./.venv/bin/python analysis/ewma_experiment.py

Writes:
    analysis/ewma_experiment.json   (to_dict(rep) for the ewma_momentum variant)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path("/Users/yahirbahena/GitHub/F1 Predicitor Project/backend")
sys.path.insert(0, str(BACKEND))

import config  # noqa: E402
from ml.features import FEATURE_COLS, _with_championship_points  # noqa: E402
from ml.evaluate import (  # noqa: E402
    BASELINE_JSON_PATH,
    run_walk_forward,
    evaluate,
    to_dict,
    render_comparison,
)

NEW_COLS = [
    "ewma_finish_3",
    "ewma_pace_3",
    "team_ewma_finish_3",
    "team_ewma_pace_3",
    "pace_trend_5",
    "finish_trend_5",
    "team_pace_trend_5",
]
OUT_JSON = BACKEND / "analysis" / "ewma_experiment.json"


# ---------------------------------------------------------------------------
# Independent recomputations for the leakage check (do NOT reuse ml.features
# helpers -- these are a from-scratch second implementation).
# ---------------------------------------------------------------------------
def manual_ewma(vals: list[float], halflife: float) -> float:
    v = [x for x in vals if not (x is None or np.isnan(x))]
    if not v:
        return np.nan
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    n = len(v)
    w = [(1.0 - alpha) ** (n - 1 - i) for i in range(n)]  # adjust=True, newest weight 1
    return sum(wi * vi for wi, vi in zip(w, v)) / sum(w)


def manual_slope(vals: list[float], n: int = 5) -> float:
    v = [x for x in vals if not (x is None or np.isnan(x))][-n:]
    if len(v) < 3:
        return np.nan
    k = len(v)
    x = list(range(k))
    xm, ym = sum(x) / k, sum(v) / k
    den = sum((xi - xm) ** 2 for xi in x)
    return sum((xi - xm) * (yi - ym) for xi, yi in zip(x, v)) / den


def _race_mean_series(t: pd.DataFrame, col: str) -> list[float]:
    """Per-race team mean of `col`, chronological (mirrors driver_features)."""
    if not len(t):
        return []
    g = t.groupby(["season", "round"], sort=True)[col].mean()
    return g.tolist()


def leakage_check(results: pd.DataFrame, races: pd.DataFrame, df: pd.DataFrame,
                  samples: list[tuple[int, int, str]]) -> str:
    """For each (season, round, driver_code) sample, rebuild history from ONLY
    strictly-prior races and recompute each new feature by hand; assert equality
    with the parquet value. Returns a human-readable report."""
    races = races.sort_values(["season", "round"]).reset_index(drop=True)
    results = _with_championship_points(results).sort_values(["season", "round"]).reset_index(drop=True)
    lines = ["Leakage check (hand-recompute from strictly-prior races only):"]
    all_ok = True
    for season, rnd, code in samples:
        i = races.index[(races["season"] == season) & (races["round"] == rnd)][0]
        prior = races.iloc[:i][["season", "round"]]
        hist = results.merge(prior, on=["season", "round"], how="inner")
        row_this = results[(results["season"] == season) & (results["round"] == rnd) &
                           (results["driver_code"] == code)].iloc[0]
        team = row_this["team"]
        d = hist[hist["driver_code"] == code].sort_values(["season", "round"])
        t = hist[hist["team"] == team].sort_values(["season", "round"])

        exp = {
            "ewma_finish_3": manual_ewma(d["finish_pos"].tolist(), 3),
            "ewma_pace_3": manual_ewma(d["pace_delta_ms"].tolist(), 3),
            "team_ewma_finish_3": manual_ewma(_race_mean_series(t, "finish_pos"), 3),
            "team_ewma_pace_3": manual_ewma(_race_mean_series(t, "pace_delta_ms"), 3),
            "pace_trend_5": manual_slope(d["pace_delta_ms"].tolist(), 5),
            "finish_trend_5": manual_slope(d["finish_pos"].tolist(), 5),
            "team_pace_trend_5": manual_slope(_race_mean_series(t, "pace_delta_ms"), 5),
        }
        got = df[(df["season"] == season) & (df["round"] == rnd) &
                 (df["driver_code"] == code)].iloc[0]
        lines.append(f"  {season} R{rnd} {code} (team={team}, {len(d)} prior driver races):")
        for c in NEW_COLS:
            e, g = exp[c], float(got[c])
            ok = (np.isnan(e) and np.isnan(g)) or np.isclose(e, g, rtol=1e-9, atol=1e-9)
            all_ok &= ok
            es = "NaN" if np.isnan(e) else f"{e:.6f}"
            gs = "NaN" if np.isnan(g) else f"{g:.6f}"
            lines.append(f"    {c:20s} hand={es:>12s}  parquet={gs:>12s}  {'OK' if ok else 'MISMATCH'}")
    lines.append(f"  => {'ALL MATCH' if all_ok else 'MISMATCH FOUND'}")
    return "\n".join(lines), all_ok


# ---------------------------------------------------------------------------
def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset.parquet")
    results = pd.read_parquet(config.RESULTS_PARQUET)
    races = pd.read_parquet(config.RACES_PARQUET)

    # ---- Leakage check on 3 sampled rows (need enough history) ----
    samples = [(2024, 12, "VER"), (2025, 5, "NOR"), (2023, 20, "HAM")]
    report, leak_ok = leakage_check(results, races, df, samples)
    print(report)
    print()

    # ---- Standard-framework evaluation of the variant ----
    feature_cols = FEATURE_COLS + NEW_COLS
    print(f"Running walk-forward: {len(feature_cols)} features "
          f"({len(FEATURE_COLS)} baseline + {len(NEW_COLS)} new)...")
    races_wf = run_walk_forward(df, feature_cols)
    rep = evaluate(races_wf)  # default blend_w = production BLEND_W (0.55)
    rep_dict = to_dict(rep)
    OUT_JSON.write_text(json.dumps(rep_dict, indent=2))
    print(f"[written] {OUT_JSON}\n")

    baseline = json.loads(BASELINE_JSON_PATH.read_text())
    print(render_comparison(baseline, [("ewma_momentum", rep_dict)]))
    print()

    # ---- Blend sensitivity (cheap: no refit) ----
    print("## Blend sensitivity (variant, no refit)\n")
    print("| w | winner_hit_rate | winner_hits | podium | mae |")
    print("|---|---|---|---|---|")
    for w in (0.55, 0.70, 0.85, 1.00):
        r = evaluate(races_wf, blend_w=w)
        o = r.overall
        print(f"| {w:.2f} | {o['winner_hit_rate']:.4f} | {o['winner_hits']} | "
              f"{o['podium_overlap_avg']:.4f} | {o['position_mae']:.4f} |")
    print()

    # ---- 2024-season delta + flipped races (variant @0.55 vs baseline @0.55) ----
    base_season = {r["group"]: r for r in baseline["breakdowns"]["by_season"]}
    var_season = {r["group"]: r for r in rep_dict["breakdowns"]["by_season"]}
    print("## Per-season winner_hit (baseline -> ewma_momentum, both w=0.55)\n")
    for yr in sorted(var_season):
        b, v = base_season.get(yr), var_season[yr]
        bstr = f"{b['winner_hit_rate']:.4f} ({b['hits']}/{b['n']})" if b else "n/a"
        print(f"  {yr}: {bstr} -> {v['winner_hit_rate']:.4f} ({v['hits']}/{v['n']})  "
              f"delta {v['winner_hit_rate'] - (b['winner_hit_rate'] if b else 0):+.4f}")
    print()

    # ---- Adoption rule ----
    o = rep.overall
    yr24 = var_season.get("2024")
    b24 = base_season.get("2024")
    wh, pod, mae = o["winner_hit_rate"], o["podium_overlap_avg"], o["position_mae"]
    passes = (wh > 0.6479 or (abs(wh - 0.6479) < 1e-9 and yr24 and b24 and
                              yr24["winner_hit_rate"] > b24["winner_hit_rate"])) \
        and wh >= 0.6479 and pod >= 2.05 and mae <= 3.33
    print("## Adoption rule @ w=0.55\n")
    print(f"  winner_hit {wh:.4f} >= 0.6479 : {wh >= 0.6479}")
    print(f"  podium     {pod:.4f} >= 2.05   : {pod >= 2.05}")
    print(f"  mae        {mae:.4f} <= 3.33   : {mae <= 3.33}")
    print(f"  leakage check passed          : {leak_ok}")
    print(f"  => DECISION: {'ADOPT' if passes and leak_ok else 'DO NOT ADOPT'}")


if __name__ == "__main__":
    main()
