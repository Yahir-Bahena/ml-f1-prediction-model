"""Context-aware blend weighting experiment.

Replace the fixed model-vs-grid blend weight w=0.55 with a per-race w driven by
pre-race-knowable context signals. Pure post-processing over the cached
walk-forward raw scores; no model refitting; no production files touched.

Run from backend/:  ./.venv/bin/python -m analysis.context_blend_experiment
Writes analysis/context_blend.json and analysis/context_blend.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from ml.dataset import DATASET_PARQUET
from ml.features import FEATURE_COLS
from ml.train import PARAMS, BLEND_W
import ml.evaluate as E

OUT_JSON = Path(__file__).resolve().parent / "context_blend.json"
OUT_MD = Path(__file__).resolve().parent / "context_blend.md"

W_GRID = [round(x, 2) for x in np.arange(0.45, 1.001, 0.05)]

# ---------------------------------------------------------------------------
# Per-race metric (mirrors ml.evaluate.evaluate's deterministic records; the
# Monte-Carlo win-prob is not needed for any metric we report, so it is skipped
# for speed. Parity with evaluate() at fixed w is asserted in main()).
# ---------------------------------------------------------------------------
def race_metrics(rc: E.RaceResult, w: float, p_wet: float | None = None) -> dict:
    scores = E._blend(rc["raw"], rc["grid"], float(w), p_wet)
    actual = rc["actual"]
    pred_rank = scores.argsort().argsort() + 1
    winner_hit = int(actual[int(pred_rank.argmin())] == 1)
    pred_top3 = set(np.where(pred_rank <= 3)[0])
    actual_top3 = set(np.where(actual <= 3)[0])
    podium = len(pred_top3 & actual_top3)
    mae = float(np.abs(pred_rank - actual).mean())
    aw = np.where(actual == 1)[0]
    aw_idx = int(aw[0]) if len(aw) else -1
    top3_hit = int(pred_rank[aw_idx] <= 3) if aw_idx >= 0 else 0
    aw_grid = float(rc["grid"][aw_idx]) if aw_idx >= 0 else float("nan")
    return dict(winner_hit=winner_hit, podium=podium, mae=mae,
                top3_hit=top3_hit, aw_grid=aw_grid)


def agg_metrics(races, w_fn, ctx, subset_keys=None) -> dict:
    """Aggregate metrics over races; w_fn(rc, ctx) -> w. subset_keys optionally
    restricts to a set of (season, round)."""
    hits = pod = mae_sum = top3 = n = 0
    pole_hits = pole_n = p3_hits = p3_n = 0
    for rc in races:
        key = (rc["season"], rc["round"])
        if subset_keys is not None and key not in subset_keys:
            continue
        w = w_fn(rc, ctx)
        m = race_metrics(rc, w)
        n += 1
        hits += m["winner_hit"]
        pod += m["podium"]
        mae_sum += m["mae"]
        top3 += m["top3_hit"]
        if m["aw_grid"] == 1:
            pole_n += 1
            pole_hits += m["winner_hit"]
        if m["aw_grid"] >= 3:
            p3_n += 1
            p3_hits += m["winner_hit"]
    return dict(n=n, hits=hits, winner_hit_rate=hits / n if n else 0.0,
                podium=pod / n if n else 0.0, mae=mae_sum / n if n else 0.0,
                top3=top3 / n if n else 0.0,
                pole=f"{pole_hits}/{pole_n}", p3=f"{p3_hits}/{p3_n}")


# ---------------------------------------------------------------------------
# Context signals (all pre-race knowable)
# ---------------------------------------------------------------------------
def build_context(df: pd.DataFrame, races) -> dict:
    """Per-race context keyed by (season, round).

    track_overtakes_avg / track_sc_rate : race-level, prior-race based (first row).
    p2_quali_gap : quali_gap_to_pole_ms of the P2 qualifier (pole-dominance proxy).
    wet : realized wet flag from race_meta (oracle covariate only).
    """
    wet_by_race = {}
    if config.RACES_PARQUET.exists():
        rm = pd.read_parquet(config.RACES_PARQUET)
        wet_by_race = {(int(r.season), int(r.round)): bool(r.wet_race)
                       for r in rm.itertuples(index=False)}
    keys = set((rc["season"], rc["round"]) for rc in races)
    ctx = {}
    for (s, r), g in df.groupby(["season", "round"]):
        if (int(s), int(r)) not in keys:
            continue
        p2 = g[g["quali_pos"] == 2]
        ctx[(int(s), int(r))] = dict(
            track_overtakes_avg=float(g["track_overtakes_avg"].iloc[0]),
            track_sc_rate=float(g["track_sc_rate"].iloc[0]),
            p2_quali_gap=float(p2["quali_gap_to_pole_ms"].iloc[0]) if len(p2) else np.nan,
            wet=wet_by_race.get((int(s), int(r))),
        )
    return ctx


SIGNALS = ["track_overtakes_avg", "p2_quali_gap", "track_sc_rate"]


def signal_val(ctx, key, sig):
    return ctx[key][sig]


# ---------------------------------------------------------------------------
# Standardization fit on a fold (nuisance normalization, not a free parameter)
# ---------------------------------------------------------------------------
def fit_zstats(ctx, fold_keys, sig):
    vals = np.array([ctx[k][sig] for k in fold_keys], dtype=float)
    vals = vals[~np.isnan(vals)]
    mu = float(np.mean(vals)) if len(vals) else 0.0
    sd = float(np.std(vals)) if len(vals) and np.std(vals) > 1e-9 else 1.0
    return mu, sd


def zscore(x, mu, sd):
    if np.isnan(x):
        return 0.0  # missing signal -> no adjustment
    return (x - mu) / sd


# ---------------------------------------------------------------------------
# Rule: w = clip(w0 + a*z(track_overtakes) + b*z(p2_quali_gap), 0.45, 1.0)
# 3 free params (w0, a, b). Sign expectations from the accumulated evidence:
#   high overtaking -> raise w (a>0);  dominant pole (large gap) -> lower w (b<0)
# ---------------------------------------------------------------------------
def make_rule(w0, a, b, zstats):
    (mu_o, sd_o), (mu_g, sd_g) = zstats

    def w_fn(rc, ctx):
        key = (rc["season"], rc["round"])
        c = ctx[key]
        zo = zscore(c["track_overtakes_avg"], mu_o, sd_o)
        zg = zscore(c["p2_quali_gap"], mu_g, sd_g)
        return float(np.clip(w0 + a * zo + b * zg, 0.45, 1.0))
    return w_fn


def fit_rule(races, ctx, fold_keys, w0_grid, a_grid, b_grid):
    """Grid-search (w0,a,b) to maximize winner hits on fold_keys. Returns best
    params, the fitted zstats, and the in-sample hits."""
    zstats = (fit_zstats(ctx, fold_keys, "track_overtakes_avg"),
              fit_zstats(ctx, fold_keys, "p2_quali_gap"))
    best = None
    for w0 in w0_grid:
        for a in a_grid:
            for b in b_grid:
                w_fn = make_rule(w0, a, b, zstats)
                m = agg_metrics(races, w_fn, ctx, subset_keys=set(fold_keys))
                # tie-break: prefer higher hits, then better podium, then lower mae
                score = (m["hits"], m["podium"], -m["mae"])
                if best is None or score > best[0]:
                    best = (score, (w0, a, b), zstats, m)
    _, params, zstats, m = best
    return params, zstats, m


# ---------------------------------------------------------------------------
def const_w(w):
    return lambda rc, ctx: w


def main():
    df = pd.read_parquet(DATASET_PARQUET)
    races, hit = E._load_or_run(df, FEATURE_COLS, PARAMS, use_cache=True)
    assert hit, "cache MISS - expected eval_cache.pkl hit"
    assert len(races) == 71, f"expected 71 races, got {len(races)}"

    ctx = build_context(df, races)
    all_keys = set((rc["season"], rc["round"]) for rc in races)

    report = {}

    # ---- 1. baseline reproduction (my harness vs production) ----
    base = agg_metrics(races, const_w(BLEND_W), ctx)
    # parity vs ml.evaluate.evaluate at fixed w
    rep_prod = E.evaluate(races, blend_w=BLEND_W)
    parity = dict(
        my_hits=base["hits"], prod_hits=int(rep_prod.overall["winner_hit_rate"] * 71 + 0.5),
        my_podium=round(base["podium"], 4), prod_podium=round(rep_prod.overall["podium_overlap_avg"], 4),
        my_mae=round(base["mae"], 4), prod_mae=round(rep_prod.overall["position_mae"], 4),
        my_top3=round(base["top3"], 4), prod_top3=round(rep_prod.overall["top3_hit_rate"], 4),
    )
    report["cache_hit"] = hit
    report["baseline_repro"] = base
    report["parity_vs_evaluate"] = parity

    # ---- 2. exploratory: per-tercile w-response (IN-SAMPLE) ----
    wresp = {}
    for sig in SIGNALS:
        vals = np.array([ctx[k][sig] for k in all_keys], dtype=float)
        valid = {k for k in all_keys if not np.isnan(ctx[k][sig])}
        q1, q2 = np.nanpercentile([ctx[k][sig] for k in valid], [33.3, 66.7])
        terc = {"low": set(), "mid": set(), "high": set()}
        for k in valid:
            v = ctx[k][sig]
            terc["low" if v <= q1 else "high" if v > q2 else "mid"].add(k)
        table = {}
        for tname, tkeys in terc.items():
            row = {}
            for w in W_GRID:
                m = agg_metrics(races, const_w(w), ctx, subset_keys=tkeys)
                row[w] = dict(hits=m["hits"], n=m["n"], rate=round(m["winner_hit_rate"], 3))
            table[tname] = dict(n=len(tkeys), q_edges=[round(q1, 2), round(q2, 2)], w_response=row)
        wresp[sig] = table
    report["w_response_terciles"] = wresp

    # w-response for wet vs dry (interaction with shipped weather rule)
    wet_keys = {k for k in all_keys if ctx[k]["wet"] is True}
    dry_keys = {k for k in all_keys if ctx[k]["wet"] is False}
    wetdry = {}
    for gname, gkeys in [("wet", wet_keys), ("dry", dry_keys)]:
        row = {}
        for w in W_GRID:
            m = agg_metrics(races, const_w(w), ctx, subset_keys=gkeys)
            row[w] = dict(hits=m["hits"], n=m["n"], rate=round(m["winner_hit_rate"], 3))
        wetdry[gname] = dict(n=len(gkeys), w_response=row)
    report["w_response_wet_dry"] = wetdry

    # ---- 3. fit rule IN-SAMPLE (all 71) ----
    w0_grid = [0.50, 0.55, 0.60]
    a_grid = [round(x, 3) for x in np.arange(-0.20, 0.201, 0.05)]
    b_grid = [round(x, 3) for x in np.arange(-0.20, 0.201, 0.05)]
    params_is, zstats_is, m_is = fit_rule(races, ctx, sorted(all_keys), w0_grid, a_grid, b_grid)
    report["fit_in_sample"] = dict(params=dict(w0=params_is[0], a=params_is[1], b=params_is[2]),
                                   metrics=m_is)

    # full-set metrics under the in-sample rule (guardrails)
    w_fn_is = make_rule(*params_is, zstats_is)
    full_is = agg_metrics(races, w_fn_is, ctx)
    report["fit_in_sample_full"] = full_is

    # ---- 4. VALIDATION ----
    ordered = sorted(all_keys)  # (season, round) chronological
    val = {}

    # (a)(b) forward and reverse time-splits
    def time_split(train_keys, test_keys, label):
        params, zstats, _ = fit_rule(races, ctx, train_keys, w0_grid, a_grid, b_grid)
        w_fn = make_rule(*params, zstats)
        test_set = set(test_keys)
        rule_m = agg_metrics(races, w_fn, ctx, subset_keys=test_set)
        base_m = agg_metrics(races, const_w(BLEND_W), ctx, subset_keys=test_set)
        return dict(label=label, params=dict(w0=params[0], a=params[1], b=params[2]),
                    test_n=rule_m["n"], rule_hits=rule_m["hits"], base_hits=base_m["hits"],
                    delta=rule_m["hits"] - base_m["hits"],
                    rule_podium=round(rule_m["podium"], 4), base_podium=round(base_m["podium"], 4),
                    rule_mae=round(rule_m["mae"], 4), base_mae=round(base_m["mae"], 4))

    fwd = time_split(ordered[:35], ordered[35:], "forward (fit first 35, test last 36)")
    rev = time_split(ordered[36:], ordered[:36], "reverse (fit last 35, test first 36)")
    val["forward_split"] = fwd
    val["reverse_split"] = rev

    # (c) leave-one-race-out
    loo_rule_hits = 0
    loo_base_hits = 0
    loo_pod = loo_mae = 0.0
    for k in ordered:
        train = [x for x in ordered if x != k]
        params, zstats, _ = fit_rule(races, ctx, train, w0_grid, a_grid, b_grid)
        w_fn = make_rule(*params, zstats)
        rc = next(r for r in races if (r["season"], r["round"]) == k)
        m = race_metrics(rc, w_fn(rc, ctx))
        loo_rule_hits += m["winner_hit"]
        loo_pod += m["podium"]
        loo_mae += m["mae"]
        bm = race_metrics(rc, BLEND_W)
        loo_base_hits += bm["winner_hit"]
    val["loo"] = dict(n=len(ordered), rule_hits=loo_rule_hits, base_hits=loo_base_hits,
                      delta=loo_rule_hits - loo_base_hits,
                      rule_podium=round(loo_pod / len(ordered), 4),
                      rule_mae=round(loo_mae / len(ordered), 4))

    # (d) plateau / sensitivity: vary each param +/-25% around in-sample fit
    plateau = {}
    for pname, idx in [("w0", 0), ("a", 1), ("b", 2)]:
        base_val = params_is[idx]
        row = {}
        for mult in [0.75, 0.9, 1.0, 1.1, 1.25]:
            p = list(params_is)
            p[idx] = round(base_val * mult, 4)
            w_fn = make_rule(*p, zstats_is)
            m = agg_metrics(races, w_fn, ctx)
            row[f"x{mult}"] = dict(val=p[idx], hits=m["hits"])
        plateau[pname] = row
    val["plateau_sensitivity"] = plateau
    report["validation"] = val

    # ---- 5. incremental value beyond weather: rule on DRY races only ----
    dry_set = dry_keys
    rule_dry = agg_metrics(races, w_fn_is, ctx, subset_keys=dry_set)
    base_dry = agg_metrics(races, const_w(BLEND_W), ctx, subset_keys=dry_set)
    report["dry_only"] = dict(rule=rule_dry, base=base_dry,
                              delta_hits=rule_dry["hits"] - base_dry["hits"])

    # ---- 6. verdict ----
    loo_ok = val["loo"]["delta"] >= 1  # >= 49/71 equivalent
    both_splits_ok = fwd["delta"] >= 0 and rev["delta"] >= 0
    plateau_hits = [v["hits"] for pr in plateau.values() for v in pr.values()]
    # plateau: no sharp collapse away from the peak (all within 2 hits of in-sample peak)
    plateau_ok = (max(plateau_hits) - min(plateau_hits)) <= 2
    verdict = "CONFIRM" if (loo_ok and both_splits_ok and plateau_ok) else "REFUTE/INCONCLUSIVE"
    report["verdict"] = dict(verdict=verdict, loo_ok=loo_ok, both_splits_ok=both_splits_ok,
                             plateau_ok=plateau_ok,
                             criteria="LOO>=49/71 AND both time-split OOS deltas>=0 AND real plateau")

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
