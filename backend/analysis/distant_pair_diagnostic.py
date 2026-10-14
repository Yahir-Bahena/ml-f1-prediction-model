"""Distant-pair diagnostic: why is the model wrong on 6+-position-margin pairs?

Wrong distant pairs are never variance — something systematic causes each one.
But pair counts amplify single events (one shock weekend creates up to ~15
wrong pairs), so the analysis attributes each wrong pair to its CULPRIT (the
driver whose own rank error is larger), collapses pairs into driver-race
events, and categorizes those events.

Run from backend/:  PYTHONPATH=. ./.venv/bin/python analysis/distant_pair_diagnostic.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from ml.dataset import DATASET_PARQUET
from ml.evaluate import _blend, _load_or_run
from ml.features import FEATURE_COLS
from ml.train import BLEND_W, PARAMS

MARGIN = 6


def main() -> None:
    ds = pd.read_parquet(DATASET_PARQUET)
    races, _ = _load_or_run(ds, FEATURE_COLS, PARAMS, use_cache=True)
    meta = pd.read_parquet(config.RACES_PARQUET).set_index(["season", "round"])
    dmeta = ds.set_index(["season", "round", "driver_code"])[["team", "quali_pos"]]

    pairs = []
    for rc in races:
        s = _blend(rc["raw"], rc["grid"], BLEND_W)
        pred = s.argsort().argsort() + 1
        actual, codes = rc["actual"], rc["driver_codes"]
        err = pred - actual  # + = finished worse than predicted? no: pred - actual; err<0 -> finished worse than predicted rank? careful below
        n = len(codes)
        for i in range(n):
            for j in range(i + 1, n):
                if rc["dnf"][i] or rc["dnf"][j]:
                    continue
                if abs(actual[i] - actual[j]) < MARGIN:
                    continue
                if (pred[i] - pred[j]) * (actual[i] - actual[j]) > 0:
                    continue
                # culprit = larger own |pred - actual|
                ci = i if abs(pred[i] - actual[i]) >= abs(pred[j] - actual[j]) else j
                pairs.append({
                    "season": rc["season"], "round": rc["round"],
                    "race": f"{rc['season']} R{rc['round']:>2} {rc['event_name'].replace(' Grand Prix', '')}",
                    "culprit": codes[ci],
                    "pred": int(pred[ci]), "finish": int(actual[ci]),
                    "grid": float(rc["grid"][ci]),
                    "direction": "collapse" if actual[ci] > pred[ci] else "surge",
                })
    d = pd.DataFrame(pairs)
    print(f"wrong distant pairs (clean, margin >= {MARGIN}): {len(d)}")

    # collapse into driver-race events
    ev = (d.groupby(["season", "round", "race", "culprit", "pred", "finish", "grid", "direction"])
            .size().rename("pairs").reset_index()
            .sort_values("pairs", ascending=False))
    print(f"unique culprit driver-race events: {len(ev)}")
    print(f"pairs from top 10 events: {ev['pairs'].head(10).sum()} "
          f"({ev['pairs'].head(10).sum() / len(d):.0%} of all wrong distant pairs)")

    ev = ev.join(meta[["wet_race", "sc_laps"]], on=["season", "round"])
    qp = [dmeta.loc[(r.season, r["round"], r.culprit)]["quali_pos"]
          if (r.season, r["round"], r.culprit) in dmeta.index else np.nan
          for _, r in ev.iterrows()]
    ev["quali"] = qp

    print("\ntop 15 events by wrong pairs generated:")
    print(ev.head(15)[["race", "culprit", "quali", "pred", "finish", "direction",
                       "wet_race", "sc_laps", "pairs"]].to_string(index=False))

    print("\ndirection split (events):", dict(ev["direction"].value_counts()))
    print("wet-race share of events:", f"{ev['wet_race'].mean():.0%}",
          "| wet share of all races: 23%")
    print("SC>=5-laps share of events:", f"{(ev['sc_laps'] >= 5).mean():.0%}")

    # categorize events
    def cat(r) -> str:
        if r.direction == "collapse":
            if not np.isnan(r.quali) and r.quali <= 8 and r.finish >= 12:
                return "collapse: qualified well, race went wrong (incident/strategy/damage)"
            return "collapse: model over-ranked a weak weekend"
        if not np.isnan(r.quali) and r.quali >= 11 and r.finish <= 8:
            return "surge: comeback drive from low grid"
        return "surge: model under-ranked a strong weekend"

    ev["category"] = ev.apply(cat, axis=1)
    print("\nevent categories (events / wrong pairs generated):")
    for c, g in ev.groupby("category"):
        print(f"  {c:<62} {len(g):>3} events, {g['pairs'].sum():>4} pairs")

    print("\nmost frequent culprit drivers (events):")
    print(ev.groupby("culprit").agg(events=("pairs", "size"), pairs=("pairs", "sum"))
            .sort_values("pairs", ascending=False).head(10).to_string())

    per_season = ev.groupby("season").agg(events=("pairs", "size"), pairs=("pairs", "sum"))
    print("\nper season:")
    print(per_season.to_string())


if __name__ == "__main__":
    main()
