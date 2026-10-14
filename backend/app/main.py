"""F1 Race Predictor API.

Run from backend/:  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
from app.services import predictor, stats, store, updater
from ml.train import METRICS_PATH

app = FastAPI(title="F1 Race Predictor", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _catch_up_on_startup() -> None:
    # a race may have finished while the server (or the whole machine) was down;
    # kick off a background ingest+retrain if the dataset is behind. Non-blocking.
    updater.maybe_catch_up()


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "data_ready": config.RESULTS_PARQUET.exists(),
        "model_ready": predictor.MODEL_PATH.exists(),
    }


@app.get("/api/next-race")
def next_race() -> dict:
    updater.maybe_catch_up()  # background ingest+retrain if a finished race is missing
    event = store.next_event()
    if event is None:
        raise HTTPException(404, "No upcoming race found for the current season")
    sessions = []
    for i in range(1, 6):
        name, date = event.get(f"Session{i}"), event.get(f"Session{i}DateUtc")
        if name and not pd.isna(date):
            sessions.append({"name": str(name), "start_utc": store.iso_utc(date)})
    return store.clean(
        {
            "season": config.CURRENT_SEASON,
            "round": int(event["RoundNumber"]),
            "name": event["EventName"],
            "official_name": event["OfficialEventName"],
            "location": config.normalize_location(event["Location"]),
            "country": config.normalize_country(
                config.normalize_location(event["Location"]), event["Country"]
            ),
            "race_start_utc": store.iso_utc(event["Session5DateUtc"]),
            "sessions": sessions,
        }
    )


@app.get("/api/predictions/next")
def predictions_next() -> dict:
    updater.maybe_catch_up()  # background ingest+retrain if a finished race is missing
    if not predictor.MODEL_PATH.exists():
        raise HTTPException(503, "Model not trained yet - run: python -m ml.train")
    result = predictor.predict_next_race()
    if result is None:
        raise HTTPException(404, "No upcoming race to predict")
    return result


@app.get("/api/model/metrics")
def model_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise HTTPException(404, "No metrics yet - run: python -m ml.train")
    return json.loads(METRICS_PATH.read_text())


@app.get("/api/schedule")
def schedule(season: int = config.CURRENT_SEASON) -> dict:
    sched = store.schedule(season)
    races = store.races()
    ingested = set(zip(races["season"], races["round"]))
    events = []
    for _, ev in sched.iterrows():
        events.append(
            {
                "round": int(ev["RoundNumber"]),
                "name": ev["EventName"],
                "location": config.normalize_location(ev["Location"]),
                "country": config.normalize_country(
                    config.normalize_location(ev["Location"]), ev["Country"]
                ),
                "race_start_utc": store.iso_utc(ev["Session5DateUtc"]),
                "completed": (season, int(ev["RoundNumber"])) in ingested,
            }
        )
    return store.clean({"season": season, "events": events})


@app.get("/api/drivers")
def drivers() -> list[dict]:
    return stats.driver_list()


@app.get("/api/drivers/{code}/stats")
def driver_stats(code: str) -> dict:
    result = stats.driver_stats(code.upper())
    if result is None:
        raise HTTPException(404, f"Unknown driver code: {code}")
    return result


@app.get("/api/tracks")
def tracks() -> list[dict]:
    return stats.track_list()


@app.get("/api/tracks/{location}/stats")
def track_stats(location: str) -> dict:
    result = stats.track_stats(location)
    if result is None:
        raise HTTPException(404, f"Unknown track location: {location}")
    return result


@app.get("/api/standings")
def standings() -> dict:
    return stats.standings()


@app.get("/api/results")
def results_list() -> list[dict]:
    """Summary of every past race the model predicted (newest first)."""
    return [
        {
            "season": r["season"], "round": r["round"],
            "event_name": r["event_name"], "location": r["location"],
            "country": r["country"], "date": r["date"], "wet_race": r["wet_race"],
            "predicted_winner": r["predicted_winner"], "actual_winner": r["actual_winner"],
            "metrics": r["metrics"], "n_highlights": len(r["highlights"]),
        }
        for r in store.predictions_history()
    ]


@app.get("/api/results/{season}/{rnd}")
def results_detail(season: int, rnd: int) -> dict:
    """Full predicted-vs-actual comparison + race-control highlights for one race."""
    for r in store.predictions_history():
        if r["season"] == season and r["round"] == rnd:
            return r
    raise HTTPException(404, f"No prediction record for {season} round {rnd}")
