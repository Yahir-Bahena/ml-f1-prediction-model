"""Reusable evaluation framework for the F1 race-winner model.

Every experiment (new features, tuned params, alternative blends) is scored here
so results are comparable byte-for-byte. The framework is split into two halves
that mirror the production pipeline but never modify it:

- ``run_walk_forward`` reproduces ``ml.train``'s walk-forward backtest exactly
  (same fit, labels, warmup, and skip rules) and returns the per-race raw ranker
  scores. It is the ONLY expensive step, so its output is cached and reused.
- ``evaluate`` is pure post-processing over those cached scores: it applies the
  production blend (imported from ``ml.train`` -- never re-derived), runs the
  Monte Carlo simulation for win probabilities, and rolls everything up into an
  ``EvalReport`` with overall metrics plus fixed breakdowns.

Design choices worth flagging:
- The blend math and per-race metric come straight from ``ml.train``
  (``blend_scores`` / ``_race_eval``) so a run at the production feature set and
  ``w = BLEND_W`` reproduces ``metrics.json`` to within floating point.
- ``simulate_race`` already takes a ``seed`` (default 42); we pass it explicitly
  so win-probability numbers are reproducible without touching global RNG state.
- Per-driver DNF probabilities use the exact formula ``app.services.predictor``
  ships with: ``clip(0.5*driver_dnf_rate + 0.5*team_dnf_rate, 0.02, 0.35)`` with
  NaN rates filled to 0.1. Races whose rows lack those columns fall back to a
  flat 0.1 rate (documented on the ``dnf_rates`` argument).

Run from backend/:  python -m ml.evaluate  [--no-cache]
Writes data/models/evaluation.md and evaluation.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, TypedDict

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

import config
from ml.dataset import DATASET_PARQUET
from ml.features import FEATURE_COLS
# Import the production blend + per-race metric so evaluation cannot drift from
# training. MAX_LABEL / PARAMS / BLEND_W define the fit that must be reproduced.
from ml.train import (
    BLEND_W,
    MAX_LABEL,
    PARAMS,
    WET_BLEND_THRESHOLD,
    _race_eval,
    blend_scores,
)
from ml.simulate import simulate_race

MODEL_PATH = config.MODELS_DIR / "model.joblib"
EVAL_CACHE_PATH = config.MODELS_DIR / "eval_cache.pkl"
EVAL_MD_PATH = config.MODELS_DIR / "evaluation.md"
EVAL_JSON_PATH = config.MODELS_DIR / "evaluation.json"
# Frozen reference every experiment is compared against (--save-baseline /
# --compare). Overwriting it is a deliberate act, never a side effect of a run.
BASELINE_JSON_PATH = config.MODELS_DIR / "baseline.json"
BASELINE_MD_PATH = config.MODELS_DIR / "baseline.md"

# Simulation config. n_sims matches the production predictor; the seed is passed
# explicitly to simulate_race for reproducible win probabilities.
SIM_N_SIMS = 10_000
SIM_SEED = 42
# Flat fallback DNF rate when driver/team reliability columns are unavailable
# (mirrors predictor.py's fillna(0.1) before the 0.5/0.5 blend).
FALLBACK_DNF_RATE = 0.1


class RaceResult(TypedDict):
    """One walk-forward test race: everything needed to score it post-hoc.

    Arrays are aligned to the test race's row order. ``grid`` is the NaN-filled
    grid actually fed to the blend (matches ml.train); ``dnf`` / ``*_dnf_rate``
    feed the Monte Carlo simulation.
    """

    season: int
    round: int
    event_name: str
    location: str
    n_features: int
    raw: np.ndarray            # raw ranker output, higher = better
    grid: np.ndarray           # NaN-filled grid used by the blend
    actual: np.ndarray         # actual finishing positions (target_pos)
    driver_codes: list[str]
    dnf: np.ndarray            # per-driver DNF flag (bool)
    driver_dnf_rate: np.ndarray
    team_dnf_rate: np.ndarray


@dataclass
class EvalReport:
    """Rolled-up evaluation: overall metrics, fixed breakdowns, and run meta."""

    overall: dict
    breakdowns: dict[str, list[dict]]
    blend_w: float
    sigma: float
    n_races: int
    n_features: Optional[int] = None
    model_path: Optional[str] = None
    # per-race records kept for callers that want to slice further; excluded
    # from to_dict/JSON to keep the artifact small.
    races_df: Optional[pd.DataFrame] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Walk-forward (the only expensive step)
# ---------------------------------------------------------------------------
def _fit(train: pd.DataFrame, feature_cols: list[str], params: dict) -> LGBMRanker:
    """Fit the lambdarank ranker with one group per race (mirrors ml.train._fit,
    but parameterized on feature list + params so experiments can plug in)."""
    train = train.sort_values(["season", "round"], kind="stable")
    groups = train.groupby(["season", "round"], sort=False).size().to_numpy()
    labels = (MAX_LABEL - train["target_pos"]).clip(lower=0).astype(int)
    model = LGBMRanker(**params)
    model.fit(train[feature_cols], labels, group=groups)
    return model


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_train_races: int = 30,
    params: Optional[dict] = None,
) -> list[RaceResult]:
    """Reproduce ml.train's walk-forward backtest and return per-race raw scores.

    Identical to the training backtest: first ``min_train_races`` races are
    train-only warmup, each later race is predicted from all strictly-prior
    races, and races with a NaN target or fewer than 10 rows are skipped. Works
    with any ``feature_cols`` / ``params`` so variants reuse the same harness.
    """
    params = params or PARAMS
    race_keys = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    race_keys = list(race_keys.itertuples(index=False, name=None))

    # ordinal race index -> fast train-subset masking (equivalent to ml.train's
    # per-row membership test, without the O(n) apply)
    ord_map = {k: i for i, k in enumerate(race_keys)}
    df = df.copy()
    df["_ord"] = list(zip(df["season"], df["round"]))
    df["_ord"] = df["_ord"].map(ord_map)

    has_dnf_cols = {"driver_dnf_rate", "team_dnf_rate"}.issubset(df.columns)
    n_features = len(feature_cols)

    out: list[RaceResult] = []
    for i in range(min_train_races, len(race_keys)):
        season, rnd = race_keys[i]
        train = df[df["_ord"] < i]
        test = df[(df["season"] == season) & (df["round"] == rnd)]
        if test["target_pos"].isna().any() or len(test) < 10:
            continue

        model = _fit(train, feature_cols, params)
        grid = test["grid_pos"].fillna(len(test)).to_numpy().astype(float)
        raw = model.predict(test[feature_cols])
        n = len(test)
        out.append(
            RaceResult(
                season=int(season),
                round=int(rnd),
                event_name=str(test.iloc[0]["event_name"]),
                location=str(test.iloc[0]["location"]),
                n_features=n_features,
                raw=np.asarray(raw, dtype=float),
                grid=grid,
                actual=test["target_pos"].to_numpy().astype(float),
                driver_codes=test["driver_code"].tolist(),
                dnf=test["dnf"].to_numpy().astype(bool),
                driver_dnf_rate=(
                    test["driver_dnf_rate"].to_numpy(dtype=float)
                    if has_dnf_cols
                    else np.full(n, np.nan)
                ),
                team_dnf_rate=(
                    test["team_dnf_rate"].to_numpy(dtype=float)
                    if has_dnf_cols
                    else np.full(n, np.nan)
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Post-processing / metrics
# ---------------------------------------------------------------------------
def _dnf_probs(
    rc: RaceResult,
    dnf_rates: Optional[Callable[[RaceResult], np.ndarray] | float],
) -> np.ndarray:
    """Per-driver DNF probability for the simulation.

    Default (dnf_rates is None): the production formula
    ``clip(0.5*driver_dnf_rate + 0.5*team_dnf_rate, 0.02, 0.35)`` with NaN rates
    filled to 0.1. A float overrides with a flat rate; a callable returns a
    per-driver array. Races whose walk-forward rows lacked the reliability
    columns carry all-NaN rates, which fill to the flat 0.1 fallback.
    """
    n = len(rc["raw"])
    if callable(dnf_rates):
        return np.asarray(dnf_rates(rc), dtype=float)
    if isinstance(dnf_rates, (int, float)):
        return np.full(n, float(dnf_rates))
    driver = np.where(np.isnan(rc["driver_dnf_rate"]), FALLBACK_DNF_RATE, rc["driver_dnf_rate"])
    team = np.where(np.isnan(rc["team_dnf_rate"]), FALLBACK_DNF_RATE, rc["team_dnf_rate"])
    return np.clip(0.5 * driver + 0.5 * team, 0.02, 0.35)


def _blend(
    raw: np.ndarray, grid: np.ndarray, w: float, p_wet: Optional[float] = None
) -> np.ndarray:
    """General-w version of ml.train.blend_scores (same math, parameterized on w
    so experiments can sweep blend weights). Parity with the production function
    at w=BLEND_W is asserted on every evaluate() call; if ml.train's blend logic
    changes, that assertion fails loudly instead of this copy silently drifting."""
    model_rank = (-raw).argsort().argsort() + 1.0
    grid = np.asarray(grid, dtype=float)
    if np.all(np.isnan(grid)):
        return model_rank
    if p_wet is not None and not np.isnan(p_wet) and p_wet >= WET_BLEND_THRESHOLD:
        w = w + (1.0 - w) * float(np.clip(p_wet, 0.0, 1.0))
    grid = np.where(np.isnan(grid), float(len(grid)), grid)
    grid_rank = grid.argsort().argsort() + 1.0
    return (1.0 - w) * grid_rank + w * model_rank + 1e-6 * grid_rank


def _assert_blend_parity(rc: RaceResult) -> None:
    """Guarantee _blend(w=BLEND_W) is byte-identical to ml.train.blend_scores."""
    for p_wet in (None, 0.3, 0.9):
        ours = _blend(rc["raw"], rc["grid"], BLEND_W, p_wet)
        prod = blend_scores(rc["raw"], rc["grid"], p_wet)
        if not np.array_equal(ours, prod):
            raise RuntimeError(
                "ml.evaluate._blend no longer matches ml.train.blend_scores "
                f"(p_wet={p_wet}); update _blend to mirror the production blend."
            )


def _agg(sub: pd.DataFrame) -> dict:
    """Minimum breakdown block: hit rates, podium overlap, MAE, spearman, n."""
    return {
        "n": int(len(sub)),
        "winner_hit_rate": float(sub["winner_hit"].mean()) if len(sub) else 0.0,
        "top3_hit_rate": float(sub["top3_hit"].mean()) if len(sub) else 0.0,
        "podium_overlap": float(sub["podium_overlap"].mean()) if len(sub) else 0.0,
        "mae": float(sub["mae"].mean()) if len(sub) else 0.0,
        "spearman": float(sub["spearman"].mean()) if len(sub) else 0.0,
        "hits": int(sub["winner_hit"].sum()) if len(sub) else 0,
    }


def evaluate(
    races: list[RaceResult],
    blend_w: float = BLEND_W,
    p_wet_fn: Optional[Callable[[RaceResult], Optional[float]]] = None,
    sigma: Optional[float] = None,
    dnf_rates: Optional[Callable[[RaceResult], np.ndarray] | float] = None,
) -> EvalReport:
    """Score cached walk-forward races. Pure post-processing (no model fitting).

    blend_w    : model-vs-grid weight; default BLEND_W reproduces production
                 (parity with ml.train.blend_scores is asserted every call).
    p_wet_fn   : optional rain-probability hook, p_wet_fn(rc) -> p_wet in [0, 1]
                 or None; lets experiments test the wet-conditional blend. When
                 None the dry-weather blend is used (reproduces production).
    sigma      : Monte Carlo noise; defaults to the trained bundle's sigma.
    dnf_rates  : see _dnf_probs (None -> production formula).
    """
    if not races:
        raise ValueError("evaluate() got no races; run_walk_forward first")

    if sigma is None:
        sigma = float(joblib.load(MODEL_PATH)["sigma"])
        model_path: Optional[str] = str(MODEL_PATH)
    else:
        model_path = None

    # race_meta wet flag for the wet/dry breakdown (optional; skipped if absent)
    wet_by_race: dict[tuple[int, int], bool] = {}
    if config.RACES_PARQUET.exists():
        rm = pd.read_parquet(config.RACES_PARQUET)
        wet_by_race = {
            (int(r.season), int(r.round)): bool(r.wet_race)
            for r in rm.itertuples(index=False)
        }

    _assert_blend_parity(races[0])

    records = []
    for rc in races:
        p_wet = p_wet_fn(rc) if p_wet_fn is not None else None
        scores = _blend(rc["raw"], rc["grid"], float(blend_w), p_wet)
        actual = rc["actual"]

        ev = _race_eval(scores, actual)  # winner_hit, podium_overlap, mae, spearman
        ev.pop("residuals", None)

        pred_rank = scores.argsort().argsort() + 1  # 1 = predicted winner
        # clean pairwise accuracy: % of non-DNF driver pairs ordered correctly
        ok = ~rc["dnf"]
        pr, ac = pred_rank[ok], actual[ok]
        diff = np.sign(np.subtract.outer(pr, pr)) * np.sign(np.subtract.outer(ac, ac))
        n_pairs = ok.sum() * (ok.sum() - 1) / 2
        pairwise_acc = float((diff > 0).sum() / 2 / n_pairs) if n_pairs else np.nan
        pw_idx = int(pred_rank.argmin())
        aw_hits = np.where(actual == 1)[0]
        aw_idx = int(aw_hits[0]) if len(aw_hits) else -1
        top3_hit = int(pred_rank[aw_idx] <= 3) if aw_idx >= 0 else 0
        aw_grid = float(rc["grid"][aw_idx]) if aw_idx >= 0 else float("nan")

        # win probabilities via the same Monte Carlo the app uses
        sim = simulate_race(scores, _dnf_probs(rc, dnf_rates), sigma,
                            n_sims=SIM_N_SIMS, seed=SIM_SEED)
        win = sim["win_prob"]

        records.append(
            {
                "season": rc["season"],
                "round": rc["round"],
                "event_name": rc["event_name"],
                "location": rc["location"],
                "winner_hit": ev["winner_hit"],
                "top3_hit": top3_hit,
                "pairwise_acc": pairwise_acc,
                "podium_overlap": ev["podium_overlap"],
                "mae": ev["mae"],
                "spearman": ev["spearman"],
                "pw_dnf": bool(rc["dnf"][pw_idx]),
                "aw_grid": aw_grid,
                "wet": wet_by_race.get((rc["season"], rc["round"])),
                "pred_winner_prob": float(win[pw_idx]),
                "actual_winner_prob": float(win[aw_idx]) if aw_idx >= 0 else float("nan"),
            }
        )

    d = pd.DataFrame(records)
    nodnf = d[~d["pw_dnf"]]

    overall = {
        "n_races": int(len(d)),
        "blend_w": float(blend_w),
        "winner_hit_rate": float(d["winner_hit"].mean()),
        "winner_hits": f"{int(d['winner_hit'].sum())}/{len(d)}",
        "top3_hit_rate": float(d["top3_hit"].mean()),
        "pairwise_accuracy": float(d["pairwise_acc"].mean()),
        "podium_overlap_avg": float(d["podium_overlap"].mean()),
        "position_mae": float(d["mae"].mean()),
        "spearman_avg": float(d["spearman"].mean()),
        "dnf_adjusted_winner_hit_rate": float(nodnf["winner_hit"].mean()) if len(nodnf) else 0.0,
        "dnf_adjusted_winner_hits": f"{int(nodnf['winner_hit'].sum())}/{len(nodnf)}",
        "avg_predicted_winner_prob": float(d["pred_winner_prob"].mean()),
        "avg_actual_winner_prob": float(d["actual_winner_prob"].mean()),
    }

    breakdowns = _build_breakdowns(d)

    return EvalReport(
        overall=overall,
        breakdowns=breakdowns,
        blend_w=float(blend_w),
        sigma=float(sigma),
        n_races=int(len(d)),
        n_features=races[0].get("n_features"),
        model_path=model_path,
        races_df=d,
    )


def _build_breakdowns(d: pd.DataFrame) -> dict[str, list[dict]]:
    """Fixed breakdown tables, each a list of labelled group rows."""
    breakdowns: dict[str, list[dict]] = {}

    # by season
    breakdowns["by_season"] = [
        {"group": str(int(yr)), **_agg(d[d["season"] == yr])}
        for yr in sorted(d["season"].unique())
    ]

    # wet vs dry (only if the wet flag joined)
    if d["wet"].notna().any():
        breakdowns["wet_vs_dry"] = [
            {"group": "wet", **_agg(d[d["wet"] == True])},   # noqa: E712
            {"group": "dry", **_agg(d[d["wet"] == False])},  # noqa: E712
        ]

    # actual winner started on pole vs not
    breakdowns["winner_pole_vs_not"] = [
        {"group": "pole", **_agg(d[d["aw_grid"] == 1])},
        {"group": "not_pole", **_agg(d[d["aw_grid"] != 1])},
    ]

    # actual winner started P3+ (grid >= 3) vs front-row
    breakdowns["winner_grid_p3plus"] = [
        {"group": "grid>=3", **_agg(d[d["aw_grid"] >= 3])},
        {"group": "grid<=2", **_agg(d[d["aw_grid"] < 3])},
    ]
    return breakdowns


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def to_dict(report: EvalReport) -> dict:
    """JSON-safe dict of the report (per-race frame intentionally omitted)."""
    return {
        "blend_w": report.blend_w,
        "sigma": report.sigma,
        "n_races": report.n_races,
        "n_features": report.n_features,
        "model_path": report.model_path,
        "overall": report.overall,
        "breakdowns": report.breakdowns,
    }


def _breakdown_table(title: str, rows: list[dict]) -> str:
    lines = [
        f"### {title}",
        "",
        "| group | n | winner_hit | top3_hit | podium | mae | spearman |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['group']} | {r['n']} | {r['winner_hit_rate']:.4f} ({r['hits']}/{r['n']}) | "
            f"{r['top3_hit_rate']:.4f} | {r['podium_overlap']:.4f} | "
            f"{r['mae']:.4f} | {r['spearman']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_markdown(report: EvalReport) -> str:
    """Readable markdown report: header, overall block, and every breakdown."""
    o = report.overall
    feat = f"{report.n_features} features" if report.n_features else "features n/a"
    model = report.model_path or "(sigma overridden)"

    lines = [
        "# F1 Race-Winner Model - Evaluation Report",
        "",
        f"- blend weight (w): {report.blend_w}",
        f"- sigma (Monte Carlo noise): {report.sigma:.4f}",
        f"- eval races: {report.n_races}",
        f"- model: {model}  ({feat})",
        f"- win-prob sim: {SIM_N_SIMS} draws, seed {SIM_SEED}",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---|",
        f"| winner_hit_rate | {o['winner_hit_rate']:.4f} ({o['winner_hits']}) |",
        f"| top3_hit_rate | {o['top3_hit_rate']:.4f} |",
        f"| pairwise_accuracy | {o['pairwise_accuracy']:.4f} |",
        f"| podium_overlap_avg | {o['podium_overlap_avg']:.4f} |",
        f"| position_mae | {o['position_mae']:.4f} |",
        f"| spearman_avg | {o['spearman_avg']:.4f} |",
        f"| dnf_adjusted_winner_hit_rate | {o['dnf_adjusted_winner_hit_rate']:.4f} "
        f"({o['dnf_adjusted_winner_hits']}) |",
        f"| avg_predicted_winner_prob | {o['avg_predicted_winner_prob']:.4f} |",
        f"| avg_actual_winner_prob | {o['avg_actual_winner_prob']:.4f} |",
        "",
        "## Breakdowns",
        "",
    ]

    titles = {
        "by_season": "By season",
        "wet_vs_dry": "Wet vs dry",
        "winner_pole_vs_not": "Actual winner: pole vs not",
        "winner_grid_p3plus": "Actual winner: grid P3+ vs front-row",
    }
    for key, rows in report.breakdowns.items():
        lines.append(_breakdown_table(titles.get(key, key), rows))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Baseline / comparison
# ---------------------------------------------------------------------------
# Flat metric rows shown in the comparison table: (label, path into to_dict()).
# Breakdown slices are addressed as (breakdown_key, group_label).
_COMPARE_ROWS: list[tuple[str, str]] = [
    ("winner_hit_rate", "winner_hit_rate"),
    ("top3_hit_rate", "top3_hit_rate"),
    ("pairwise_accuracy", "pairwise_accuracy"),
    ("podium_overlap_avg", "podium_overlap_avg"),
    ("position_mae", "position_mae"),
    ("spearman_avg", "spearman_avg"),
    ("dnf_adjusted_winner_hit_rate", "dnf_adjusted_winner_hit_rate"),
    ("avg_predicted_winner_prob", "avg_predicted_winner_prob"),
]
_COMPARE_SLICES: list[tuple[str, str, str]] = [
    ("wet winner_hit", "wet_vs_dry", "wet"),
    ("dry winner_hit", "wet_vs_dry", "dry"),
    ("pole-winner hit", "winner_pole_vs_not", "pole"),
    ("P3+-winner hit", "winner_grid_p3plus", "grid>=3"),
]
# position_mae is the only metric where lower is better.
_LOWER_IS_BETTER = {"position_mae"}


def _flat_metrics(rep: dict) -> dict[str, float]:
    """Flatten a to_dict() report into the comparison rows (label -> value).
    Skips metrics absent from older reports (e.g. pre-pairwise baselines)."""
    out = {label: rep["overall"][key] for label, key in _COMPARE_ROWS
           if key in rep["overall"]}
    for label, bkey, group in _COMPARE_SLICES:
        rows = {r["group"]: r for r in rep.get("breakdowns", {}).get(bkey, [])}
        if group in rows:
            out[label] = rows[group]["winner_hit_rate"]
    for row in rep.get("breakdowns", {}).get("by_season", []):
        out[f"{row['group']} winner_hit"] = row["winner_hit_rate"]
    return out


def render_comparison(baseline: dict, experiments: list[tuple[str, dict]]) -> str:
    """Markdown table: baseline column, one column per experiment, with deltas.

    `baseline` / experiment dicts are to_dict() reports (e.g. loaded from
    baseline.json / an experiment's evaluation.json).
    """
    base = _flat_metrics(baseline)
    exps = [(name, _flat_metrics(rep)) for name, rep in experiments]

    header = "| metric | baseline |" + "".join(f" {n} | delta |" for n, _ in exps)
    sep = "|---|---|" + "---|---|" * len(exps)
    lines = [
        "# Experiment comparison",
        "",
        f"Baseline: {baseline.get('n_races')} races, w={baseline.get('blend_w')}, "
        f"{baseline.get('n_features')} features.",
        "",
        header,
        sep,
    ]
    for label, base_v in base.items():
        cells = [f"| {label} | {base_v:.4f} |"]
        for _, ex in exps:
            v = ex.get(label)
            if v is None:
                cells.append(" - | - |")
                continue
            delta = v - base_v
            if label in _LOWER_IS_BETTER:
                delta = -delta  # report improvement as positive
            cells.append(f" {v:.4f} | {delta:+.4f} |")
        lines.append("".join(cells))
    lines += ["", "_delta > 0 = better than baseline (MAE sign flipped accordingly)._", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI / cache
# ---------------------------------------------------------------------------
def _cache_key(feature_cols: list[str], params: dict, n_rows: int) -> str:
    """Stable hash of the inputs that determine the walk-forward output. Any
    change to the feature list, params, or dataset size invalidates the cache."""
    payload = json.dumps(
        {"features": list(feature_cols), "params": {k: params[k] for k in sorted(params)},
         "n_rows": n_rows},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_or_run(
    df: pd.DataFrame, feature_cols: list[str], params: dict, use_cache: bool
) -> tuple[list[RaceResult], bool]:
    """Return (races, cache_hit). Caches raw walk-forward scores keyed by
    _cache_key so re-runs are instant and stale caches auto-invalidate."""
    key = _cache_key(feature_cols, params, len(df))
    if use_cache and EVAL_CACHE_PATH.exists():
        payload = joblib.load(EVAL_CACHE_PATH)
        if payload.get("key") == key:
            return payload["races"], True
    races = run_walk_forward(df, feature_cols, params=params)
    joblib.dump({"key": key, "races": races}, EVAL_CACHE_PATH)
    return races, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the production F1 model setup.")
    parser.add_argument("--no-cache", action="store_true",
                        help="force a fresh walk-forward refit (ignore eval_cache.pkl)")
    parser.add_argument("--save-baseline", action="store_true",
                        help="freeze this run as the reference baseline "
                             "(baseline.json/.md; overwrites any previous baseline)")
    parser.add_argument("--compare", nargs="+", metavar="NAME=REPORT.json",
                        help="compare experiment report(s) against the saved baseline "
                             "instead of evaluating; e.g. --compare quali_surprise=exp.json")
    args = parser.parse_args()

    config.ensure_dirs()

    if args.compare:
        if not BASELINE_JSON_PATH.exists():
            raise SystemExit("no saved baseline - run `python -m ml.evaluate --save-baseline` first")
        baseline = json.loads(BASELINE_JSON_PATH.read_text())
        experiments = []
        for spec in args.compare:
            name, _, path = spec.rpartition("=")
            name = name or Path(path).stem
            experiments.append((name, json.loads(Path(path).read_text())))
        print(render_comparison(baseline, experiments))
        return

    df = pd.read_parquet(DATASET_PARQUET)

    t0 = time.perf_counter()
    races, cache_hit = _load_or_run(df, FEATURE_COLS, PARAMS, use_cache=not args.no_cache)
    t_wf = time.perf_counter() - t0
    print(f"Walk-forward: {len(races)} races "
          f"({'cache hit' if cache_hit else 'refit'}, {t_wf:.2f}s)")

    report = evaluate(races, blend_w=BLEND_W)
    md = render_markdown(report)
    print("\n" + md)

    EVAL_MD_PATH.write_text(md)
    EVAL_JSON_PATH.write_text(json.dumps(to_dict(report), indent=2))
    print(f"\nSaved {EVAL_MD_PATH} and {EVAL_JSON_PATH}")

    if args.save_baseline:
        BASELINE_JSON_PATH.write_text(json.dumps(to_dict(report), indent=2))
        BASELINE_MD_PATH.write_text(md.replace(
            "# F1 Race-Winner Model - Evaluation Report",
            "# F1 Race-Winner Model - BASELINE (frozen reference)", 1))
        print(f"Baseline frozen: {BASELINE_JSON_PATH} and {BASELINE_MD_PATH}")


if __name__ == "__main__":
    main()
