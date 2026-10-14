"""Phase 2: evaluate the practice long-run features vs the frozen 48/71 baseline.

Consumes analysis/practice_features.csv (from practice_features_extract.py).
Builds the three MVP features, merges into the dataset, runs the standard
walk-forward framework, and reports: comparison vs baseline, blend sweep,
target-race scorecard (the 6 'practice could fix' misses), flips, per-season.

F2 practice_pace_surprise = driver delta_ms MINUS the team's rolling mean of its
own practice deltas over the last 5 weekends-with-data (shift(1) — strictly
prior). Practice-space norms: stable team fuel/program bias cancels (the VER
sprint-FP1 false-negative defense).

Run from backend/:  ./.venv/bin/python analysis/practice_pace_experiment.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ml.dataset import DATASET_PARQUET
from ml.evaluate import evaluate, render_comparison, run_walk_forward, to_dict
from ml.features import FEATURE_COLS

TARGETS = [  # (season, round, label, winner)
    (2024, 6, "Miami 24", "NOR"), (2024, 19, "Austin 24", "LEC"),
    (2025, 5, "Jeddah 25", "PIA"), (2026, 9, "Silverstone 26", "LEC"),
    (2025, 6, "Miami 25", "PIA"), (2025, 22, "Vegas 25", "VER"),
]
NEW = ["practice_longrun_delta_ms", "practice_pace_surprise", "practice_sample_confidence"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    pf = pd.read_csv("analysis/practice_features.csv")
    pf = pf[pf["status"] == "OK"][["season", "round", "driver", "delta_ms", "confidence"]]
    df = df.merge(pf, left_on=["season", "round", "driver_code"],
                  right_on=["season", "round", "driver"], how="left").drop(columns=["driver"])
    df = df.rename(columns={"delta_ms": "practice_longrun_delta_ms",
                            "confidence": "practice_sample_confidence"})
    df["practice_sample_confidence"] = df["practice_sample_confidence"].fillna(0)

    # team rolling practice norm (strictly prior weekends-with-data)
    df = df.sort_values(["season", "round"]).reset_index(drop=True)
    tw = (df.groupby(["team", "season", "round"], sort=False)["practice_longrun_delta_ms"]
            .mean().rename("team_wk").reset_index())
    tw = tw.sort_values(["season", "round"])
    tw["team_norm"] = (tw.groupby("team")["team_wk"]
                         .transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean()))
    df = df.merge(tw[["team", "season", "round", "team_norm"]], on=["team", "season", "round"], how="left")
    df["practice_pace_surprise"] = df["practice_longrun_delta_ms"] - df["team_norm"]
    return df.drop(columns=["team_norm"])


def main() -> None:
    df = build_features(pd.read_parquet(DATASET_PARQUET))
    cov = df["practice_longrun_delta_ms"].notna()
    print(f"feature coverage: {cov.mean():.1%} of driver-races "
          f"({df[cov].groupby(['season','round']).ngroups} weekends with data)")
    print(f"surprise coverage: {df['practice_pace_surprise'].notna().mean():.1%}")

    baseline = json.load(open("data/models/baseline.json"))
    variants = {
        "practice_full": NEW,
        "practice_delta_only": ["practice_longrun_delta_ms", "practice_sample_confidence"],
    }
    reports = {}
    for name, cols in variants.items():
        races = run_walk_forward(df, FEATURE_COLS + cols)
        reports[name] = (races, evaluate(races))
    print(render_comparison(baseline, [(n, to_dict(rep)) for n, (_, rep) in reports.items()]))

    for name, (races, _) in reports.items():
        print(f"\n=== {name}: blend sweep ===")
        for w in (0.55, 0.7, 0.85):
            o = evaluate(races, blend_w=w).overall
            print(f"  w={w}: {o['winner_hits']} | pod {o['podium_overlap_avg']:.3f} | mae {o['position_mae']:.3f}")

    # target scorecard + flips for the full variant
    from ml.evaluate import _blend
    races, rep = reports["practice_full"]
    base_hits = {(r["season"], r["round"]): None for r in []}
    print("\n=== target scorecard (practice_full, w=0.55) ===")
    per_race = {}
    for rc in races:
        s = _blend(rc["raw"], rc["grid"], 0.55)
        pw = int((s.argsort().argsort() + 1).argmin())
        aw = int(np.where(rc["actual"] == 1)[0][0])
        per_race[(rc["season"], rc["round"])] = (rc["driver_codes"][pw], rc["driver_codes"][aw], pw == aw)
    for season, rnd, label, winner in TARGETS:
        pick, aw, hit = per_race.get((season, rnd), ("?", "?", False))
        print(f"  {label:<15} winner {winner}: pick {pick} -> {'HIT' if hit else 'miss'}")

    d = rep.races_df
    json.dump(to_dict(rep), open("analysis/practice_full.json", "w"), indent=2)
    json.dump(to_dict(reports["practice_delta_only"][1]),
              open("analysis/practice_delta_only.json", "w"), indent=2)
    print("\nsaved analysis/practice_full.json + practice_delta_only.json")


if __name__ == "__main__":
    main()
