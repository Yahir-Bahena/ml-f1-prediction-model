"""Experiment E2: margin-aware override gating for the F1 race-winner model.

Post-processing only -- no model fitting. Reads cached walk-forward raw scores
from analysis/raw_scores_cache_pace.pkl (variant "baseline" ONLY) and tests
whether a raw-score MARGIN between the blend's pick and the grid leader
(pole-sitter) can gate ("revert to pole") the 5 known wrong overrides without
killing the 10 correct ones.

Experiment only -- does NOT touch production ml/*.py or app/**.

Run from backend/ with the project venv:
    ./.venv/bin/python analysis/margin_gate_experiment.py

Writes:
    analysis/margin_gate_results.md
"""
from __future__ import annotations
import pickle, io
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BACKEND = Path("/Users/yahirbahena/GitHub/F1 Predicitor Project/backend")
ANALYSIS = BACKEND / "analysis"
CACHE = ANALYSIS / "raw_scores_cache_pace.pkl"
RESULTS_MD = ANALYSIS / "margin_gate_results.md"

W = 0.55  # production blend weight


# ---------------------------------------------------------------------------
# Production blend (verbatim from ml/train.py logic; lower score = better)
# ---------------------------------------------------------------------------
def blend(raw, grid, w=W):
    model_rank = (-raw).argsort().argsort() + 1.0   # 1 = model favourite (raw high)
    grid_rank = grid.argsort().argsort() + 1.0      # 1 = pole
    return (1.0 - w) * grid_rank + w * model_rank + 1e-6 * grid_rank


def softmax(x, T):
    z = (x - x.max()) / T
    e = np.exp(z)
    return e / e.sum()


MARGIN_NAMES = ["m1", "m2", "m3_T1", "m3_Tstd", "m4"]
MARGIN_DESC = {
    "m1": "(raw[pick]-raw[pole]) / std(raw)",
    "m2": "(raw[pick]-raw[pole]) / (max(raw)-min(raw))",
    "m3_T1": "softmax(raw,T=1)[pick]-softmax(raw,T=1)[pole]",
    "m3_Tstd": "softmax(raw,T=std)[pick]-softmax(raw,T=std)[pole]",
    "m4": "(raw[model#1]-raw[model#2]) / std(raw)  (general top-pick conviction)",
}


def compute_margins(raw):
    """All candidate margins for one race, given the blend pick and pole indices
    supplied by caller via closure is avoided: we return a fn."""
    std = raw.std()
    rng = raw.max() - raw.min()
    mo = (-raw).argsort()  # model order, best first
    r1, r2 = int(mo[0]), int(mo[1])
    sm1 = softmax(raw, 1.0)
    smS = softmax(raw, std if std > 0 else 1.0)

    def margins(pick, pole):
        return {
            "m1": (raw[pick] - raw[pole]) / std,
            "m2": (raw[pick] - raw[pole]) / rng,
            "m3_T1": sm1[pick] - sm1[pole],
            "m3_Tstd": smS[pick] - smS[pole],
            "m4": (raw[r1] - raw[r2]) / std,
        }
    return margins


# ---------------------------------------------------------------------------
# Baseline reproduction + override table
# ---------------------------------------------------------------------------
def build_override_table(races):
    """Return (overall_hits, n_races, override_df). override_df has one row per
    override race (blend pick != pole) with margins, category, and indices."""
    hits = 0
    rows = []
    for ri, rc in enumerate(races):
        raw = np.asarray(rc["raw"], float)
        grid = np.asarray(rc["grid"], float)
        actual = np.asarray(rc["actual"])
        s = blend(raw, grid)
        pick = int(s.argmin())
        gr = grid.argsort().argsort() + 1.0
        pole = int(gr.argmin())
        aw_mask = np.where(actual == 1)[0]
        aw = int(aw_mask[0]) if len(aw_mask) else -1
        hit = int(aw == pick)
        hits += hit
        if pick == pole:
            continue
        pole_won = int(actual[pole] == 1)
        cat = "correct" if hit else ("wrong" if pole_won else "other")
        m = compute_margins(raw)(pick, pole)
        rows.append({
            "race_idx": ri, "season": rc["season"], "round": rc["round"],
            "event_name": rc["event_name"], "location": rc["location"],
            "pick": pick, "pole": pole, "categ": cat,
            "hit": hit, "pole_won": pole_won, **m,
        })
    return hits, len(races), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gate application (TOP-PICK SWAP, not full re-rank)
# ---------------------------------------------------------------------------
def eval_gated(races, margin_name, tau):
    """Apply gate: for each override race, if margin < tau revert the override by
    SWAPPING the pick and pole ranks in the predicted ranking (top-pick swap;
    everything else unchanged). Return metrics dict over all races."""
    hits = pods = 0
    maes = []
    n = 0
    saved = lost = 0  # among wrong / correct overrides
    for rc in races:
        raw = np.asarray(rc["raw"], float)
        grid = np.asarray(rc["grid"], float)
        actual = np.asarray(rc["actual"])
        s = blend(raw, grid)
        pr = s.argsort().argsort() + 1  # 1 = predicted winner
        pick = int(pr.argmin())
        gr = grid.argsort().argsort() + 1.0
        pole = int(gr.argmin())
        aw_mask = np.where(actual == 1)[0]
        aw = int(aw_mask[0]) if len(aw_mask) else -1

        reverted = False
        if pick != pole:
            m = compute_margins(raw)(pick, pole)[margin_name]
            if m < tau:
                # top-pick swap: exchange ranks of pick and pole
                rp, rl = pr[pick], pr[pole]
                pr = pr.copy()
                pr[pick], pr[pole] = rl, rp
                reverted = True
                # bookkeeping vs baseline outcome
                if aw == pole and pick != pole:   # was wrong override, now saved
                    saved += 1
                if aw == pick:                     # was correct override, now lost
                    lost += 1

        pw = int(pr.argmin())
        hit = int(aw == pw)
        hits += hit
        pred_top3 = set(np.where(pr <= 3)[0])
        actual_top3 = set(np.where(actual <= 3)[0])
        pods += len(pred_top3 & actual_top3)
        maes.append(float(np.abs(pr - actual).mean()))
        n += 1
    return {"hits": hits, "n": n, "hit_rate": hits / n,
            "podium_overlap": float(np.mean(pods / 1)) / n if False else pods / n,
            "mae": float(np.mean(maes)),
            "saved_wrong": saved, "lost_correct": lost}


# ---------------------------------------------------------------------------
# Separation stat (AUC = P(margin_correct > margin_wrong))
# ---------------------------------------------------------------------------
def auc(correct_vals, wrong_vals):
    if not correct_vals or not wrong_vals:
        return float("nan")
    gt = sum(c > w for c, w in product(correct_vals, wrong_vals))
    eq = sum(c == w for c, w in product(correct_vals, wrong_vals))
    return (gt + 0.5 * eq) / (len(correct_vals) * len(wrong_vals))


# ---------------------------------------------------------------------------
# tau sweep over override population (hit-rate view)
# ---------------------------------------------------------------------------
def sweep_tau(odf, margin_name):
    """Sweep tau over candidate thresholds; report net effect on the 71-race hit
    rate using ONLY the override population (others neutral). Returns list of
    dicts sorted by tau."""
    corr = odf[odf["categ"] == "correct"][margin_name].to_numpy()
    wrong = odf[odf["categ"] == "wrong"][margin_name].to_numpy()
    allm = np.sort(odf[margin_name].to_numpy())
    # candidate taus = midpoints between sorted unique margins + endpoints
    uniq = np.unique(allm)
    cands = [uniq[0] - 1e-6]
    for a, b in zip(uniq[:-1], uniq[1:]):
        cands.append((a + b) / 2)
    cands.append(uniq[-1] + 1e-6)
    out = []
    base_hits = int((odf["categ"] == "correct").sum())  # override-pop hits at tau=0
    for tau in cands:
        saved = int((wrong < tau).sum())        # wrong overrides reverted -> now hit
        lost = int((corr < tau).sum())           # correct overrides reverted -> now miss
        net = saved - lost
        out.append({"tau": float(tau), "saved_wrong": saved,
                    "lost_correct": lost, "net": net,
                    "override_hits": base_hits + net})
    return out, base_hits


# ---------------------------------------------------------------------------
# Overfitting controls
# ---------------------------------------------------------------------------
def best_tau_on(odf_subset, margin_name):
    """Pick tau maximizing net (saved-lost) on a subset of override races.
    Tie-break: widest plateau -> choose midpoint of the longest maximal net run."""
    sw, _ = sweep_tau(odf_subset, margin_name) if len(odf_subset) else ([], 0)
    if not sw:
        return None, 0
    best_net = max(r["net"] for r in sw)
    # find contiguous runs achieving best_net, choose midpoint of widest run
    taus = [r["tau"] for r in sw]
    nets = [r["net"] for r in sw]
    best_run = (0, 0)
    i = 0
    while i < len(nets):
        if nets[i] == best_net:
            j = i
            while j + 1 < len(nets) and nets[j + 1] == best_net:
                j += 1
            if (j - i) >= (best_run[1] - best_run[0]):
                best_run = (i, j)
            i = j + 1
        else:
            i += 1
    lo, hi = best_run
    tau_star = (taus[lo] + taus[hi]) / 2
    return tau_star, best_net


def apply_tau_to_race(row, margin_name, tau):
    """For a single override race row, does gate at tau produce a HIT?
    correct override: hit unless reverted (margin<tau).
    wrong override: hit only if reverted (margin<tau).
    other override: never a hit."""
    m = row[margin_name]
    reverted = m < tau
    if row["categ"] == "correct":
        return 0 if reverted else 1
    if row["categ"] == "wrong":
        return 1 if reverted else 0
    return 0


def leave_one_out(odf, margin_name):
    """For each override race, pick tau on the other override races, apply to the
    held-out one. Return (loo_hits_over_overrides, per-race list)."""
    hits = 0
    detail = []
    for idx in odf.index:
        train = odf.drop(index=idx)
        tau_star, _ = best_tau_on(train, margin_name)
        h = apply_tau_to_race(odf.loc[idx], margin_name, tau_star)
        hits += h
        detail.append((odf.loc[idx]["event_name"], odf.loc[idx]["categ"], tau_star, h))
    return hits, detail


def time_split(odf, races, margin_name, split_at):
    """Choose tau* on override races with race_idx < split_at; freeze; apply to
    the rest. Report OOS override-hits before/after."""
    train = odf[odf.race_idx < split_at]
    test = odf[odf.race_idx >= split_at]
    tau_star, _ = best_tau_on(train, margin_name)
    if tau_star is None or len(test) == 0:
        return None
    base_test_hits = int((test["categ"] == "correct").sum())
    gated_test_hits = sum(apply_tau_to_race(test.loc[i], margin_name, tau_star)
                          for i in test.index)
    return {"tau_star": tau_star, "n_train_ov": len(train), "n_test_ov": len(test),
            "test_base_hits": base_test_hits, "test_gated_hits": gated_test_hits,
            "test_delta": gated_test_hits - base_test_hits}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    payload = pickle.load(open(CACHE, "rb"))
    races = payload["cache"]["baseline"]

    buf = io.StringIO()
    p = lambda *a: (print(*a), print(*a, file=buf))

    p("# Experiment E2: Margin-Aware Override Gating - Results\n")
    p("Post-processing on cached walk-forward raw scores (variant `baseline`, w=0.55).")
    p("No model fitting. Gate implemented as a **top-pick swap** (exchange pick and")
    p("pole ranks; all other ranks unchanged).\n")

    # ---- 1. Baseline reproduction + overrides ----
    hits, n, odf = build_override_table(races)
    p("## 1. Baseline reproduction\n")
    p(f"- winner_hit: **{hits}/{n}** = {hits/n:.4f}  (target 46/71 = 0.6479) "
      f"-> {'MATCH' if (hits == 46 and n == 71) else 'MISMATCH'}")
    n_ov = len(odf)
    n_corr = int((odf["categ"] == "correct").sum())
    n_wrong = int((odf["categ"] == "wrong").sum())
    n_other = int((odf["categ"] == "other").sum())
    grid_leader_won = int(odf.pole_won.sum())
    p(f"- overrides (blend pick != pole): **{n_ov}**  | correct(hit) {n_corr} | "
      f"wrong(pole won) {n_wrong} | other(neither won) {n_other}")
    p(f"- grid leader won in {grid_leader_won}/{n_ov} overrides; overriding net "
      f"{n_corr}-{grid_leader_won} = {n_corr-grid_leader_won:+d}\n")
    p("### Override races (hit / pole_won / category)\n")
    p("| season | round | event | cat | hit | pole_won |")
    p("|---|---|---|---|---|---|")
    for _, r in odf.iterrows():
        p(f"| {r.season} | {r['round']} | {r.event_name} | {r["categ"]} | {r.hit} | {r.pole_won} |")
    p("")

    # ---- 2/3. Margin values + separation ----
    p("## 2-3. Margin values and separation\n")
    for mn in MARGIN_NAMES:
        p(f"### {mn}: `{MARGIN_DESC[mn]}`\n")
        # values per category (order within each category = row order in odf)
        for cat in ["correct", "wrong", "other"]:
            sub = odf[odf["categ"] == cat]
            vals = " ".join(f"{v:.3f}" for v in sub[mn])
            p(f"- **{cat}** (n={len(sub)}): {vals}")
        cvals = odf[odf["categ"] == "correct"][mn].tolist()
        wvals = odf[odf["categ"] == "wrong"][mn].tolist()
        a = auc(cvals, wvals)
        p(f"- correct[min/med/max] = {min(cvals):.3f}/{np.median(cvals):.3f}/{max(cvals):.3f}"
          f" | wrong = {min(wvals):.3f}/{np.median(wvals):.3f}/{max(wvals):.3f}")
        p(f"- **AUC P(correct>wrong) = {a:.3f}**  (need > ~0.70 for usable separation)\n")

    aucs = {mn: auc(odf[odf["categ"] == "correct"][mn].tolist(),
                    odf[odf["categ"] == "wrong"][mn].tolist()) for mn in MARGIN_NAMES}
    best_margin = max(MARGIN_NAMES, key=lambda m: aucs[m])
    p(f"Best-separating margin by AUC: **{best_margin}** (AUC {aucs[best_margin]:.3f}).")
    p("Note on the 4 'other' overrides: reverting them changes nothing (both the")
    p("blend pick and the pole-sitter lost), so they are neutral for hit-rate and")
    p("excluded from the correct-vs-wrong AUC.\n")

    # ---- 4. tau sweep (report for every margin) ----
    p("## 4. tau sweep (hit-rate view; override population only, others neutral)\n")
    p("`override_hits` = hits among the 19 override races (baseline = 10 correct).")
    p("Full-backtest winner_hit = 46 - 10 + override_hits = 36 + override_hits.\n")
    sweep_best = {}
    for mn in MARGIN_NAMES:
        sw, base = sweep_tau(odf, mn)
        best_net = max(r["net"] for r in sw)
        # plateau: taus achieving best_net
        plateau_taus = [r["tau"] for r in sw if r["net"] == best_net]
        sweep_best[mn] = (best_net, min(plateau_taus), max(plateau_taus), sw)
        p(f"### {mn}  (best net {best_net:+d} -> full-backtest "
          f"{36 + base + best_net}/71 = {(36 + base + best_net)/71:.4f})\n")
        p("| tau | saved_wrong | lost_correct | net | override_hits |")
        p("|---|---|---|---|---|")
        for r in sw:
            mark = "  <== best" if r["net"] == best_net else ""
            p(f"| {r['tau']:.4f} | {r['saved_wrong']} | {r['lost_correct']} | "
              f"{r['net']:+d} | {r['override_hits']}{mark} |")
        p(f"\nPlateau at best net: tau in [{min(plateau_taus):.4f}, {max(plateau_taus):.4f}] "
          f"(width {max(plateau_taus)-min(plateau_taus):.4f}, "
          f"{len(plateau_taus)} candidate points).\n")

    # overall best margin/tau by net then plateau width
    def keyfn(mn):
        bn, lo, hi, _ = sweep_best[mn]
        return (bn, hi - lo)
    top_margin = max(MARGIN_NAMES, key=keyfn)
    tbn, tlo, thi, _ = sweep_best[top_margin]
    p(f"## Best gate overall: **{top_margin}**, best net **{tbn:+d}**, "
      f"plateau tau in [{tlo:.4f}, {thi:.4f}].\n")

    # ---- 5. Overfitting controls ----
    p("## 5. Overfitting controls (n=19 overrides, 5 wrong)\n")

    # metrics via full eval for a representative tau (midpoint of top plateau)
    tau_mid = (tlo + thi) / 2
    full = eval_gated(races, top_margin, tau_mid)
    base_full = eval_gated(races, top_margin, -1e9)  # tau below all -> no reverts
    p(f"### 7. Metric impact at {top_margin}, tau={tau_mid:.4f} (top-pick swap)\n")
    p(f"| metric | baseline | gated | delta |")
    p("|---|---|---|---|")
    p(f"| winner_hit | {base_full['hits']}/{base_full['n']} "
      f"({base_full['hit_rate']:.4f}) | {full['hits']}/{full['n']} "
      f"({full['hit_rate']:.4f}) | {full['hit_rate']-base_full['hit_rate']:+.4f} |")
    p(f"| podium_overlap | {base_full['podium_overlap']:.3f} | "
      f"{full['podium_overlap']:.3f} | {full['podium_overlap']-base_full['podium_overlap']:+.3f} |")
    p(f"| mae | {base_full['mae']:.3f} | {full['mae']:.3f} | "
      f"{full['mae']-base_full['mae']:+.3f} |")
    p(f"- saved_wrong {full['saved_wrong']}, lost_correct {full['lost_correct']}\n")

    # Plateau interpretation
    p("### Plateau interpretation\n")
    for mn in MARGIN_NAMES:
        bn, lo, hi, _ = sweep_best[mn]
        verdict = "WIDE (plausibly real)" if (hi - lo) > 0.05 and bn >= 2 else \
                  ("narrow/low-net (overfit-prone)" )
        p(f"- {mn}: best net {bn:+d}, plateau width {hi-lo:.4f} -> {verdict}")
    p("")

    # Leave-one-race-out on the best margin (and all margins for completeness)
    p("### Leave-one-race-out (LOO) over the 19 override races\n")
    p(f"Baseline override-hits = {n_corr}/19. Gate must beat this out-of-sample.\n")
    p("| margin | LOO override_hits | net vs baseline |")
    p("|---|---|---|")
    loo_results = {}
    for mn in MARGIN_NAMES:
        lh, det = leave_one_out(odf, mn)
        loo_results[mn] = (lh, det)
        p(f"| {mn} | {lh}/19 | {lh - n_corr:+d} |")
    p("")
    p(f"LOO detail for best margin **{top_margin}** (event | cat | tau_on_other18 | hit):\n")
    for ev, cat, tau, h in loo_results[top_margin][1]:
        p(f"- {ev[:26]:26s} | {cat:7s} | tau={tau:.4f} | hit={h}")
    p("")

    # Time-split (forward and reverse) on the best margin
    p("### Time-split validation (best margin)\n")
    # split_at chosen so ~half the 71 backtest races on each side
    split_at = 35  # race_idx 0..34 vs 35..70
    fwd = time_split(odf, races, top_margin, split_at)
    # reverse: train on race_idx >= split_at, test on < split_at
    train_rev = odf[odf.race_idx >= split_at]
    test_rev = odf[odf.race_idx < split_at]
    tau_rev, _ = best_tau_on(train_rev, top_margin)
    rev = None
    if tau_rev is not None and len(test_rev):
        base_rev = int((test_rev["categ"] == "correct").sum())
        gated_rev = sum(apply_tau_to_race(test_rev.loc[i], top_margin, tau_rev)
                        for i in test_rev.index)
        rev = {"tau_star": tau_rev, "n_test_ov": len(test_rev),
               "test_base_hits": base_rev, "test_gated_hits": gated_rev,
               "test_delta": gated_rev - base_rev}
    if fwd:
        p(f"- FORWARD  (tau* on race_idx<{split_at}={fwd['n_train_ov']} ov, apply to "
          f"{fwd['n_test_ov']} ov): tau*={fwd['tau_star']:.4f}, OOS override-hits "
          f"{fwd['test_base_hits']} -> {fwd['test_gated_hits']} "
          f"(delta {fwd['test_delta']:+d})")
    if rev:
        p(f"- REVERSE  (tau* on race_idx>={split_at}={len(train_rev)} ov, apply to "
          f"{rev['n_test_ov']} ov): tau*={rev['tau_star']:.4f}, OOS override-hits "
          f"{rev['test_base_hits']} -> {rev['test_gated_hits']} "
          f"(delta {rev['test_delta']:+d})")
    p("")

    # ---- Verdict ----
    p("## Verdict against success criteria\n")
    # CONFIRM needs: some tau plateau saves >=3 wrong losing <=1 correct (net>=+2)
    # AND LOO stays >= +1 net.
    best_net_overall = tbn
    # does the best-net config satisfy saved>=3 & lost<=1 anywhere?
    strong = False
    strong_where = None
    for mn in MARGIN_NAMES:
        sw, _ = sweep_tau(odf, mn)
        for r in sw:
            if r["saved_wrong"] >= 3 and r["lost_correct"] <= 1:
                strong = True
                strong_where = (mn, r)
                break
        if strong:
            break
    loo_best = loo_results[top_margin][0]
    loo_net = loo_best - n_corr
    auc_ok = aucs[best_margin] > 0.70

    p(f"- Separation: best AUC {aucs[best_margin]:.3f} ({best_margin}); need >~0.70 "
      f"-> {'OK' if auc_ok else 'FAIL (distributions overlap)'}")
    p(f"- Strong gate (save>=3 wrong, lose<=1 correct anywhere): "
      f"{'FOUND at '+str(strong_where[0]) if strong else 'NONE'}")
    p(f"- Best in-sample net: {best_net_overall:+d} "
      f"(full-backtest {36+n_corr+best_net_overall}/71); need >=+2 for CONFIRM")
    p(f"- LOO override-hits ({top_margin}): {loo_best}/19 (net {loo_net:+d}); "
      f"need >= +1")
    p("")
    confirm = strong and (best_net_overall >= 2) and (loo_net >= 1)
    if confirm:
        p("### CONFIRM: a margin plateau saves >=3 wrong overrides losing <=1 correct, "
          "net >=+2, and holds up under LOO.")
    else:
        reasons = []
        if not auc_ok:
            reasons.append(f"margin distributions overlap (best AUC {aucs[best_margin]:.3f} < 0.70)")
        if not strong:
            reasons.append("no tau saves >=3 wrong while losing <=1 correct "
                           "(the wrong overrides Suzuka/Montreal are HIGH-margin, high-conviction)")
        if best_net_overall < 2:
            reasons.append(f"best in-sample net only {best_net_overall:+d} (< +2)")
        if loo_net < 1:
            reasons.append(f"LOO net {loo_net:+d} (< +1)")
        p("### REFUTE: " + "; ".join(reasons) + ".")

    RESULTS_MD.write_text(buf.getvalue())
    print(f"\n[written] {RESULTS_MD}")


if __name__ == "__main__":
    main()
