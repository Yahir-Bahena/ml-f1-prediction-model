"""Experiment E4: NOISE-ROBUST ("adjusted") momentum features for the F1 winner model.

Follow-up to E3 (plain EWMA, rejected 44/71). E3's autopsy showed a single DNF or
wet result poisoned the half-life-3 window. This builds a noise-robust variant:

  adj_ewma_finish_3    EWMA(hl=3) over CLASSIFIED finishes only -- mechanical /
                       generic DNFs dropped, driver-caused crashes kept (crash is
                       form signal), wet races downweighted 0.5.
  adj_ewma_pace_3      EWMA(hl=3) over pace_delta_ms, ALL DNFs excluded, wet x0.5.
  adj_team_ewma_pace_3 team per-race mean pace, same exclusions / downweighting.
  adj_finish_trend_5   OLS slope over last 5 classified-for-finish finishes.

Two variants scored with the PROJECT-STANDARD framework (ml.evaluate):
  adj_momentum   = FEATURE_COLS + all 4
  adj_pace_only  = FEATURE_COLS + [adj_ewma_pace_3, adj_team_ewma_pace_3]

Nothing here mutates production. Adoption is gated on the strict rule at the end.

Run from backend/:  ./.venv/bin/python analysis/adj_momentum_experiment.py
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
from ml.features import FEATURE_COLS, CRASH_STATUSES, _with_championship_points  # noqa: E402
from ml.evaluate import (  # noqa: E402
    BASELINE_JSON_PATH,
    run_walk_forward,
    evaluate,
    to_dict,
    render_comparison,
    _blend,
)
from ml.train import BLEND_W  # noqa: E402

ADJ_ALL = ["adj_ewma_finish_3", "adj_ewma_pace_3", "adj_team_ewma_pace_3", "adj_finish_trend_5"]
ADJ_PACE = ["adj_ewma_pace_3", "adj_team_ewma_pace_3"]
VARIANTS = {
    "adj_momentum": ADJ_ALL,
    "adj_pace_only": ADJ_PACE,
}
ANALYSIS = BACKEND / "analysis"

# The 6 outcome-changing E3 winner-flip races (autopsy).
FLIP_RACES = [
    (2024, 20, "Mexico City GP", "LOST (E3)"),
    (2024, 23, "Qatar GP", "LOST (E3)"),
    (2025, 14, "Hungarian GP", "LOST (E3)"),
    (2025, 16, "Italian GP (Monza)", "LOST (E3, track not noise)"),
    (2025, 12, "British GP", "GAINED (E3)"),
    (2025, 23, "Qatar GP", "GAINED (E3)"),
]


# ---------------------------------------------------------------------------
# Independent from-scratch recomputations for the leakage check.
# ---------------------------------------------------------------------------
def manual_ewma_wet(values, wets, halflife) -> float:
    pairs = [(v, w) for v, w in zip(values, wets) if not (v is None or np.isnan(v))]
    if not pairs:
        return np.nan
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    n = len(pairs)
    num = den = 0.0
    for i, (v, wet) in enumerate(pairs):
        weight = ((1.0 - alpha) ** (n - 1 - i)) * (0.5 if wet else 1.0)
        num += weight * v
        den += weight
    return num / den if den else np.nan


def manual_slope(vals, n=5) -> float:
    v = [x for x in vals if not (x is None or np.isnan(x))][-n:]
    if len(v) < 3:
        return np.nan
    k = len(v)
    x = list(range(k))
    xm, ym = sum(x) / k, sum(v) / k
    den = sum((xi - xm) ** 2 for xi in x)
    return sum((xi - xm) * (yi - ym) for xi, yi in zip(x, v)) / den


def plain_ewma(vals, halflife) -> float:
    v = [x for x in vals if not (x is None or np.isnan(x))]
    if not v:
        return np.nan
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    n = len(v)
    w = [(1.0 - alpha) ** (n - 1 - i) for i in range(n)]
    return sum(wi * vi for wi, vi in zip(w, v)) / sum(w)


def hand_features(d: pd.DataFrame, t: pd.DataFrame) -> dict:
    """Recompute the 4 adj features from a driver's/team's strictly-prior rows."""
    d = d.sort_values(["season", "round"])
    dnf = d["dnf"].astype(bool)
    crash = dnf & d["status"].isin(CRASH_STATUSES)
    keep_fin = d[(~dnf) | crash]
    keep_pace = d[~dnf]
    # team per-race clean pace
    tn = t[~t["dnf"].astype(bool)]
    if len(tn):
        g = tn.groupby(["season", "round"], sort=True)
        tp = g["pace_delta_ms"].mean().tolist()
        tw = g["wet_race"].max().tolist()
    else:
        tp, tw = [], []
    return {
        "adj_ewma_finish_3": manual_ewma_wet(
            keep_fin["finish_pos"].tolist(), keep_fin["wet_race"].astype(bool).tolist(), 3
        ),
        "adj_ewma_pace_3": manual_ewma_wet(
            keep_pace["pace_delta_ms"].tolist(), keep_pace["wet_race"].astype(bool).tolist(), 3
        ),
        "adj_team_ewma_pace_3": manual_ewma_wet(tp, tw, 3),
        "adj_finish_trend_5": manual_slope(keep_fin["finish_pos"].tolist(), 5),
    }


def leakage_check(results, races, df, samples):
    races = races.sort_values(["season", "round"]).reset_index(drop=True)
    results = _with_championship_points(results).sort_values(["season", "round"]).reset_index(drop=True)
    lines = ["Leakage check (hand-recompute from strictly-prior races only):"]
    all_ok = True
    for season, rnd, code in samples:
        i = races.index[(races["season"] == season) & (races["round"] == rnd)][0]
        prior = races.iloc[:i][["season", "round"]]
        hist = results.merge(prior, on=["season", "round"], how="inner")
        team = results[(results["season"] == season) & (results["round"] == rnd) &
                       (results["driver_code"] == code)].iloc[0]["team"]
        d = hist[hist["driver_code"] == code].sort_values(["season", "round"])
        t = hist[hist["team"] == team].sort_values(["season", "round"])
        exp = hand_features(d, t)
        got = df[(df["season"] == season) & (df["round"] == rnd) &
                 (df["driver_code"] == code)].iloc[0]
        lines.append(f"  {season} R{rnd} {code} (team={team}, {len(d)} prior driver races):")
        for c in ADJ_ALL:
            e, g = exp[c], float(got[c])
            ok = (np.isnan(e) and np.isnan(g)) or np.isclose(e, g, rtol=1e-9, atol=1e-6)
            all_ok &= ok
            es = "NaN" if np.isnan(e) else f"{e:.6f}"
            gs = "NaN" if np.isnan(g) else f"{g:.6f}"
            lines.append(f"    {c:22s} hand={es:>13s}  parquet={gs:>13s}  {'OK' if ok else 'MISMATCH'}")
    lines.append(f"  => {'ALL MATCH' if all_ok else 'MISMATCH FOUND'}")
    return "\n".join(lines), all_ok


def sai_mexico_demo(results, races, df):
    """Explicit SAI @ 2024 R20 Mexico: plain vs adj EWMA finish, with window."""
    races = races.sort_values(["season", "round"]).reset_index(drop=True)
    results = _with_championship_points(results).sort_values(["season", "round"]).reset_index(drop=True)
    i = races.index[(races["season"] == 2024) & (races["round"] == 20)][0]
    prior = races.iloc[:i][["season", "round"]]
    hist = results.merge(prior, on=["season", "round"], how="inner")
    d = hist[hist["driver_code"] == "SAI"].sort_values(["season", "round"])
    win = d.tail(6)[["season", "round", "event_name", "finish_pos", "dnf", "wet_race", "status"]]
    lines = ["SAI @ 2024 R20 Mexico -- last 6 prior races (window that feeds the EWMA):"]
    for _, r in win.iterrows():
        tag = "  <-- DNF (status='%s') DROPPED from adj finish" % r["status"] if r["dnf"] else ""
        lines.append(f"    {int(r['season'])} R{int(r['round']):>2} {r['event_name'][:26]:26s} "
                     f"finish={r['finish_pos']:>5} wet={bool(r['wet_race'])}{tag}")
    plain = plain_ewma(d["finish_pos"].tolist(), 3)
    row = df[(df["season"] == 2024) & (df["round"] == 20) & (df["driver_code"] == "SAI")].iloc[0]
    lines.append(f"  plain ewma_finish_3 (E3, includes Azerbaijan DNF) = {plain:.4f}  "
                 f"[parquet {float(row['ewma_finish_3']):.4f}]")
    lines.append(f"  adj_ewma_finish_3   (E4, DNF dropped, wet x0.5)   = {float(row['adj_ewma_finish_3']):.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def pred_winners(races_wf, w):
    """{(season,round): (pred_winner_code, actual_winner_code, hit)} at blend w."""
    out = {}
    for rc in races_wf:
        scores = _blend(rc["raw"], rc["grid"], w)
        pred_i = int(scores.argmin())
        codes = rc["driver_codes"]
        actual = rc["actual"]
        aw = np.where(actual == 1)[0]
        aw_code = codes[int(aw[0])] if len(aw) else "?"
        pred_code = codes[pred_i]
        out[(rc["season"], rc["round"])] = (pred_code, aw_code, int(pred_code == aw_code))
    return out


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset.parquet")
    results = pd.read_parquet(config.RESULTS_PARQUET)
    races = pd.read_parquet(config.RACES_PARQUET)
    baseline = json.loads(BASELINE_JSON_PATH.read_text())

    # ---- Leakage check (SAI Mexico mandatory) ----
    samples = [(2024, 20, "SAI"), (2025, 14, "NOR"), (2024, 23, "VER")]
    report, leak_ok = leakage_check(results, races, df, samples)
    print(report + "\n")
    print(sai_mexico_demo(results, races, df) + "\n")

    # ---- Walk-forward: baseline + both variants ----
    print("Running walk-forward (baseline + 2 variants)...")
    wf = {"baseline": run_walk_forward(df, FEATURE_COLS)}
    reps = {}
    for name, cols in VARIANTS.items():
        wf[name] = run_walk_forward(df, FEATURE_COLS + cols)
        rep = evaluate(wf[name])  # w = BLEND_W (0.55)
        reps[name] = to_dict(rep)
        (ANALYSIS / f"{name}.json").write_text(json.dumps(reps[name], indent=2))
        print(f"  [written] analysis/{name}.json  ({len(FEATURE_COLS)+len(cols)} feats)")
    print()

    # ---- Comparison table (both variants vs frozen baseline) ----
    print(render_comparison(baseline, [(n, reps[n]) for n in VARIANTS]) + "\n")

    # ---- Blend sensitivity ----
    for name in VARIANTS:
        print(f"## Blend sensitivity -- {name}\n")
        print("| w | winner_hit_rate | winner_hits | podium | mae |")
        print("|---|---|---|---|---|")
        for w in (0.55, 0.70, 0.85, 1.00):
            o = evaluate(wf[name], blend_w=w).overall
            print(f"| {w:.2f} | {o['winner_hit_rate']:.4f} | {o['winner_hits']} | "
                  f"{o['podium_overlap_avg']:.4f} | {o['position_mae']:.4f} |")
        print()

    # ---- Target-race scorecard ----
    pw = {name: pred_winners(wf[name], BLEND_W) for name in ["baseline", *VARIANTS]}
    print("## Target-race scorecard (E3 flip races, w=0.55)\n")
    print("| season R | event | actual | baseline (hit) | adj_momentum (hit) | adj_pace_only (hit) | E3 |")
    print("|---|---|---|---|---|---|---|")
    for s, r, ev, e3 in FLIP_RACES:
        b = pw["baseline"].get((s, r))
        if b is None:
            print(f"| {s} R{r} | {ev} | (not in walk-forward) |")
            continue
        aw = b[1]
        cells = []
        for name in ["baseline", "adj_momentum", "adj_pace_only"]:
            p = pw[name][(s, r)]
            cells.append(f"{p[0]} ({'Y' if p[2] else 'n'})")
        print(f"| {s} R{r} | {ev} | {aw} | {cells[0]} | {cells[1]} | {cells[2]} | {e3} |")
    print()

    # ---- New flips introduced by each variant (vs baseline) ----
    for name in VARIANTS:
        flips = []
        for key in pw["baseline"]:
            bp, aw, bh = pw["baseline"][key]
            vp, _, vh = pw[name][key]
            if bp != vp:
                outcome = ("GAINED" if vh and not bh else "LOST" if bh and not vh
                           else "neutral")
                flips.append((key, aw, bp, vp, outcome))
        print(f"## All winner flips: baseline -> {name} (w=0.55)  [{len(flips)} races]\n")
        print("| season R | actual | baseline | variant | outcome |")
        print("|---|---|---|---|---|")
        for (s, r), aw, bp, vp, oc in sorted(flips):
            print(f"| {s} R{r} | {aw} | {bp} | {vp} | {oc} |")
        print()

    # ---- Adoption rule ----
    print("## Adoption rule (strict) @ w=0.55\n")
    base_season = {row["group"]: row for row in baseline["breakdowns"]["by_season"]}
    b23 = base_season.get("2023")
    print(f"baseline: winner_hit {baseline['overall']['winner_hit_rate']:.4f} "
          f"({baseline['overall']['winner_hits']}), 2023 {b23['hits']}/{b23['n']}\n")
    decision = {}
    for name in VARIANTS:
        o = reps[name]["overall"]
        vs = {row["group"]: row for row in reps[name]["breakdowns"]["by_season"]}["2023"]
        wh = o["winner_hit_rate"]
        hits = int(o["winner_hits"].split("/")[0])
        pod, mae = o["podium_overlap_avg"], o["position_mae"]
        s23 = vs["hits"]
        c_wh = hits >= 47
        c_pod = pod >= 2.05
        c_mae = mae <= 3.33
        c_23 = s23 >= 11
        ok = c_wh and c_pod and c_mae and c_23 and leak_ok
        decision[name] = (ok, hits, pod, mae, s23)
        print(f"### {name}")
        print(f"  winner_hit {wh:.4f} ({o['winner_hits']})  >=47/71 : {c_wh}")
        print(f"  podium     {pod:.4f}  >=2.05           : {c_pod}")
        print(f"  mae        {mae:.4f}  <=3.33           : {c_mae}")
        print(f"  2023       {s23}/14   >=11/14          : {c_23}")
        print(f"  leakage passed                          : {leak_ok}")
        print(f"  => {'PASSES GATE' if ok else 'FAILS GATE'}\n")

    passing = [n for n, v in decision.items() if v[0]]
    if passing:
        best = max(passing, key=lambda n: decision[n][1])
        print(f"=> ADOPT: {best} (best passing variant, {decision[best][1]}/71)")
    else:
        print("=> DO NOT ADOPT either variant. Production FEATURE_COLS untouched.")


if __name__ == "__main__":
    main()
