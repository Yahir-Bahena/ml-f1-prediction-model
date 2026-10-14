"""Access to ingested parquet tables and the fastf1 event schedule."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config

config.setup_fastf1()

_cache: dict = {}


def _load_parquet(path) -> pd.DataFrame:
    """Load a parquet with mtime-based caching so retrains/ingests are picked up."""
    mtime = path.stat().st_mtime
    key = str(path)
    if key not in _cache or _cache[key][0] != mtime:
        _cache[key] = (mtime, pd.read_parquet(path))
    return _cache[key][1]


def results() -> pd.DataFrame:
    return _load_parquet(config.RESULTS_PARQUET)


def predictions_history() -> list[dict]:
    """Predicted-vs-actual records (newest race first). Empty if not built yet."""
    import json

    path = config.PROCESSED_DIR / "predictions_history.json"
    if not path.exists():
        return []
    mtime = path.stat().st_mtime
    key = "pred_history"
    if key not in _cache or _cache[key][0] != mtime:
        races = json.loads(path.read_text()).get("races", [])
        _cache[key] = (mtime, sorted(races, key=lambda r: (r["season"], r["round"]), reverse=True))
    return _cache[key][1]


def races() -> pd.DataFrame:
    return _load_parquet(config.RACES_PARQUET)


def schedule(season: int) -> pd.DataFrame:
    import fastf1

    key = f"schedule:{season}"
    if key not in _cache:
        _cache[key] = (None, fastf1.get_event_schedule(season, include_testing=False))
    return _cache[key][1]


def now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)


def iso_utc(ts) -> str | None:
    """ISO-8601 string with an explicit UTC designator (…Z).

    fastf1 session times are tz-naive but represent UTC. A bare isoformat() omits
    the offset, so browsers (correctly, per spec) parse the string as *local*
    time — shifting the displayed time and the countdown by the viewer's offset.
    Marking it UTC fixes both. Returns None for missing/NaT input.
    """
    if ts is None or pd.isna(ts):
        return None
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.isoformat().replace("+00:00", "Z")


def next_event() -> pd.Series | None:
    """First event of the current season whose race hasn't started yet."""
    sched = schedule(config.CURRENT_SEASON)
    upcoming = sched[sched["Session5DateUtc"] > now_utc()]
    if len(upcoming) == 0:
        return None
    return upcoming.iloc[0]


def completed_quali(event: pd.Series) -> pd.DataFrame | None:
    """Qualifying results for `event` if the session already happened, else None."""
    import fastf1

    from scripts.ingest import _quali_features

    q_time = None
    for i in range(1, 6):
        if str(event.get(f"Session{i}")) == "Qualifying":
            q_time = event.get(f"Session{i}DateUtc")
            break
    if q_time is None or pd.isna(q_time) or q_time + pd.Timedelta(hours=2) > now_utc():
        return None
    try:
        quali = fastf1.get_session(int(event["EventDate"].year), int(event["RoundNumber"]), "Q")
        quali.load(laps=False, telemetry=False, weather=False, messages=False)
        qdf = _quali_features(quali)
        return qdf if len(qdf) else None
    except Exception:
        return None


def grid_penalties(event: pd.Series) -> dict[str, int]:
    """Manual grid-place penalties for `event`, from data/grid_penalties.json.

    Keyed "<season>-<round>", so an entry applies to that race and no other:
    a stale entry stops matching the moment `next_event` advances. Missing file
    or no match returns {} -> prediction behaves exactly as with no penalties.
    Only the starting-grid prior is affected; training data and the model are
    never touched. See [[grid-penalty-gap]] in project memory for the rationale.
    """
    import json

    path = config.DATA_DIR / "grid_penalties.json"
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    key = "grid_penalties"
    if key not in _cache or _cache[key][0] != mtime:
        try:
            _cache[key] = (mtime, json.loads(path.read_text()))
        except (ValueError, OSError):
            _cache[key] = (mtime, {})
    table = _cache[key][1]
    ev_key = f"{config.CURRENT_SEASON}-{int(event['RoundNumber'])}"
    raw = table.get(ev_key, {})
    return {str(k): int(v) for k, v in raw.items() if isinstance(v, (int, float)) and int(v) > 0}


def clean(obj):
    """Recursively convert numpy types and NaN to JSON-safe python values."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if pd.isna(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        # timestamps are UTC; emit with a Z so browsers don't parse them as local
        return iso_utc(obj)
    if obj is pd.NaT:
        return None
    return obj
