"""Lazy background catch-up for the auto-update pipeline.

A launchd timer runs scripts/auto_update.sh every few hours, but on a laptop that
sleeps it can miss the window right after a race. This module closes that gap: when
the API is hit and a finished race turns out to be missing from the dataset, it
spawns the same auto_update.sh in the background. The run ingests + retrains, and
the API hot-reloads the new parquet/model by mtime (see store / predictor), so a
later refresh shows the update. Non-blocking, debounced, and single-flight, so it
is safe to call on every request.
"""
from __future__ import annotations

import subprocess
import threading
import time

import pandas as pd

import config
from app.services import store

_SCRIPT = config.BACKEND_DIR / "scripts" / "auto_update.sh"
_LOG = config.DATA_DIR / "auto_update.log"
_MIN_INTERVAL = 600  # seconds; never re-spawn within this of a previous trigger

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_last_trigger = 0.0


def _completed_but_missing() -> bool:
    """True if the schedule has a race that finished (>4h ago) but isn't ingested."""
    try:
        sched = store.schedule(config.CURRENT_SEASON)
        races = store.races()
    except Exception:
        return False
    ingested = set(zip(races["season"], races["round"]))
    now = store.now_utc()
    for _, ev in sched.iterrows():
        end = ev.get("Session5DateUtc")
        if end is None or pd.isna(end):
            continue
        # a race is "done" ~4h after lights out (race + cooldown), matching ingest
        if pd.Timestamp(end) + pd.Timedelta(hours=4) < now:
            if (config.CURRENT_SEASON, int(ev["RoundNumber"])) not in ingested:
                return True
    return False


def maybe_catch_up() -> bool:
    """Spawn auto_update.sh in the background if a finished race is missing.

    Returns True if a run was launched. No-op (returns False) when one is already
    running, was launched within _MIN_INTERVAL, the script is absent, or nothing
    is pending. Never blocks the caller on the actual ingest/retrain.
    """
    global _proc, _last_trigger
    if not _SCRIPT.exists():
        return False
    with _lock:
        if _proc is not None and _proc.poll() is None:
            return False  # a catch-up is already in flight
        if time.time() - _last_trigger < _MIN_INTERVAL:
            return False  # debounce repeated requests
        if not _completed_but_missing():
            return False
        _last_trigger = time.time()
        log = open(_LOG, "a")  # child owns the handle; mirrors the launchd log
        _proc = subprocess.Popen(
            ["/bin/bash", str(_SCRIPT)],
            stdout=log,
            stderr=log,
            start_new_session=True,  # detach: survives independent of this request
        )
        return True


def is_running() -> bool:
    """Whether a background catch-up launched by this process is still going."""
    with _lock:
        return _proc is not None and _proc.poll() is None
