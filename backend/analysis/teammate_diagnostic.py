"""T1.0: margin-controlled teammate-ordering diagnostic (measurement only).

Answers, from the existing pairwise framework:
1. Is the teammate deficit real after controlling for finish margin?
2. Quali-order baseline: how often does Saturday order alone call Sunday order?
3. Ceilings: prior head-to-head oracle and hindsight season-h2h oracle.
4. Team-tier split: top-4-car teammate pairs (winner-relevant) vs the rest.

Run from backend/:  ./.venv/bin/python analysis/teammate_diagnostic.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.dataset import DATASET_PARQUET
from ml.evaluate import _blend, _load_or_run
from ml.features import FEATURE_COLS
from ml.train import BLEND_W, PARAMS


def main() -> None:
    ds = pd.read_parquet(DATASET_PARQUET)
    races, _ = _load_or_run(ds, FEATURE_COLS, PARAMS, use_cache=True)
    meta = ds.set_index(["season", "round", "driver_code"])[
        ["team", "quali_pos", "team_season_points", "target_pos", "dnf"]]

    # full history of clean same-team meetings, for h2h records
    hist = ds[~ds["dnf"]][["season", "round", "driver_code", "team", "target_pos"]]

    tm_rows, other_rows = [], []
    for rc in races:
        s = _blend(rc["raw"], rc["grid"], BLEND_W)
        pred = s.argsort().argsort() + 1
        codes, actual = rc["driver_codes"], rc["actual"]
        n = len(codes)
        info = {}
        for i in range(n):
            try:
                info[i] = meta.loc[(rc["season"], rc["round"], codes[i])]
            except KeyError:
                info[i] = None
        pts = sorted({float(v["team_season_points"]) for v in info.values()
                      if v is not None and not pd.isna(v["team_season_points"])}, reverse=True)
        top4_pts = set(pts[:4])
        for i in range(n):
            for j in range(i + 1, n):
                if rc["dnf"][i] or rc["dnf"][j]:
                    continue
                a, b = info[i], info[j]
                correct = (pred[i] - pred[j]) * (actual[i] - actual[j]) > 0
                margin = abs(int(actual[i] - actual[j]))
                row = {"correct": bool(correct), "margin": margin}
                if a is not None and b is not None and a["team"] == b["team"]:
                    row.update(
                        season=rc["season"], round=rc["round"], team=a["team"],
                        top4=float(a["team_season_points"]) in top4_pts,
                        code_i=codes[i], code_j=codes[j],
                        quali_i=a["quali_pos"], quali_j=b["quali_pos"],
                        actual_i=float(actual[i]), actual_j=float(actual[j]),
                    )
                    tm_rows.append(row)
                else:
                    other_rows.append(row)

    tm = pd.DataFrame(tm_rows)
    ot = pd.DataFrame(other_rows)
    print(f"clean teammate pairs: {len(tm)} | clean non-teammate pairs: {len(ot)}")

    def band(m: int) -> str:
        return "1-2" if m <= 2 else "3-5" if m <= 5 else "6+"

    tm["band"], ot["band"] = tm["margin"].map(band), ot["margin"].map(band)
    print("\n1) MARGIN-CONTROLLED COMPARISON")
    print(f"{'band':>6} {'tm n':>6} {'tm acc':>8} {'other n':>8} {'other acc':>10}")
    exp_acc = 0.0
    for b in ("1-2", "3-5", "6+"):
        t, o = tm[tm["band"] == b], ot[ot["band"] == b]
        print(f"{b:>6} {len(t):>6} {t['correct'].mean():>7.1%} {len(o):>8} {o['correct'].mean():>9.1%}")
        exp_acc += len(t) / len(tm) * o["correct"].mean()
    print(f"teammate margin mix: {dict(tm['band'].value_counts(normalize=True).round(2))}")
    print(f"margin-implied expectation (non-teammate rates at teammate mix): {exp_acc:.1%}")
    print(f"observed teammate accuracy: {tm['correct'].mean():.1%} "
          f"-> residual deficit: {exp_acc - tm['correct'].mean():+.1%}")

    # 2) quali-order baseline (both quali positions known)
    q = tm.dropna(subset=["quali_i", "quali_j"])
    q_correct = ((q["quali_i"] - q["quali_j"]) * (q["actual_i"] - q["actual_j"]) > 0)
    m_correct_sub = q["correct"]
    print(f"\n2) QUALI-ORDER BASELINE (n={len(q)} teammate pairs with quali)")
    print(f"quali order alone: {q_correct.mean():.1%} | model on same subset: {m_correct_sub.mean():.1%}")

    # 3) oracles: prior h2h (walk-forward honest) and hindsight season h2h
    def h2h(sub: pd.DataFrame, ci: str, cj: str) -> int:
        """+1 if ci leads meetings, -1 if cj, 0 tie/none."""
        wins_i = wins_j = 0
        for (_, _), g in sub.groupby(["season", "round"]):
            gi, gj = g[g.driver_code == ci], g[g.driver_code == cj]
            if len(gi) and len(gj):
                if float(gi.target_pos.iloc[0]) < float(gj.target_pos.iloc[0]):
                    wins_i += 1
                else:
                    wins_j += 1
        return (wins_i > wins_j) - (wins_j > wins_i)

    prior_hits = prior_n = hind_hits = hind_n = 0
    key = lambda s, r: s * 100 + r
    for _, r in tm.iterrows():
        ci, cj = r["code_i"], r["code_j"]
        pair_hist = hist[(hist.driver_code.isin([ci, cj])) & (hist.team == r["team"])]
        by_race = pair_hist.groupby(["season", "round"]).filter(lambda g: g.driver_code.nunique() == 2)
        actual_sign = 1 if r["actual_i"] < r["actual_j"] else -1
        prior = by_race[by_race.apply(lambda x: key(x.season, x["round"]) < key(r["season"], r["round"]), axis=1)]
        lead = h2h(prior, ci, cj)
        if lead != 0:
            prior_n += 1
            prior_hits += int(lead == actual_sign)
        lead_all = h2h(by_race, ci, cj)
        if lead_all != 0:
            hind_n += 1
            hind_hits += int(lead_all == actual_sign)
    print(f"\n3) ORACLE CEILINGS")
    print(f"prior-h2h oracle:    {prior_hits}/{prior_n} = {prior_hits / max(prior_n,1):.1%}")
    print(f"hindsight-h2h oracle: {hind_hits}/{hind_n} = {hind_hits / max(hind_n,1):.1%}  (upper bound)")

    # 4) team-tier split
    print(f"\n4) TEAM TIER SPLIT (model accuracy)")
    for label, g in tm.groupby(tm["top4"].map({True: "top-4 car", False: "midfield/back car"})):
        print(f"  {label:<18} n={len(g):>4}  acc={g['correct'].mean():.1%}")


if __name__ == "__main__":
    main()
