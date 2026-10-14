"""Live rain forecast for the upcoming race via Open-Meteo (free, no key).

Design notes:
- Coordinates come from a static map keyed by canonical location — circuits
  are fixed venues, and the fastf1 schedule carries no lat/lon.
- The wet signal is thresholded to a boolean because the is_wet feature was
  trained on booleans (any-rain-during-race flags); feeding it 0/1 keeps the
  serving input in-distribution. The raw probability is surfaced to the UI.
- Robustness: 5s timeout, 6h in-process cache (forecasts do not change
  minute-to-minute), and None on ANY failure so the prediction path falls
  back to the previous behaviour (is_wet = NaN). Predictions must never
  fail or hang because of weather.
"""
from __future__ import annotations

import time

import pandas as pd
import requests  # fastf1 dependency; ships certifi CAs (stdlib urllib may lack them)

import config

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_S = 5.0
CACHE_TTL_S = 6 * 3600
RACE_WINDOW_H = 3          # race start -> +3h covers any F1 race
WET_PROB_THRESHOLD = 0.5   # forecast rain probability that flags is_wet

# circuit coordinates, keyed by canonical location (config.normalize_location)
CIRCUIT_COORDS: dict[str, tuple[float, float]] = {
    "Austin": (30.1328, -97.6411),
    "Baku": (40.3725, 49.8533),
    "Barcelona": (41.5700, 2.2611),
    "Budapest": (47.5789, 19.2486),
    "Imola": (44.3439, 11.7167),
    "Jeddah": (21.6319, 39.1044),
    "Las Vegas": (36.1147, -115.1728),
    "Le Castellet": (43.2506, 5.7917),
    "Lusail": (25.4900, 51.4542),
    "Madrid": (40.4650, -3.6110),
    "Marina Bay": (1.2914, 103.8640),
    "Melbourne": (-37.8497, 144.9680),
    "Mexico City": (19.4042, -99.0907),
    "Miami": (25.9581, -80.2389),
    "Monaco": (43.7347, 7.4206),
    "Montréal": (45.5000, -73.5228),
    "Monza": (45.6156, 9.2811),
    "Sakhir": (26.0325, 50.5106),
    "Shanghai": (31.3389, 121.2200),
    "Silverstone": (52.0786, -1.0169),
    "Spa-Francorchamps": (50.4372, 5.9714),
    "Spielberg": (47.2197, 14.7647),
    "Suzuka": (34.8431, 136.5410),
    "São Paulo": (-23.7036, -46.6997),
    "Yas Island": (24.4672, 54.6031),
    "Zandvoort": (52.3888, 4.5409),
}

_cache: dict = {}


def race_rain_forecast(location: str, race_start_utc) -> dict | None:
    """Rain forecast for the race window, or None if unavailable.

    Returns {"rain_prob": 0..1|None, "precip_mm": float|None, "is_wet": 0.0|1.0,
    "window_start_utc": iso, "source": "open-meteo"}.
    """
    loc = config.normalize_location(str(location))
    coords = CIRCUIT_COORDS.get(loc)
    if coords is None or race_start_utc is None or pd.isna(race_start_utc):
        return None
    start = pd.Timestamp(race_start_utc).floor("h")
    key = (loc, start.isoformat())
    hit = _cache.get(key)
    if hit is not None and time.time() - hit[0] < CACHE_TTL_S:
        return hit[1]

    try:
        resp = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": coords[0],
                "longitude": coords[1],
                "hourly": "precipitation,precipitation_probability",
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": (start + pd.Timedelta(hours=RACE_WINDOW_H)).strftime("%Y-%m-%d"),
                "timezone": "UTC",
            },
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        hours = pd.to_datetime(data["hourly"]["time"])
        in_window = (hours >= start) & (hours <= start + pd.Timedelta(hours=RACE_WINDOW_H))
        probs = data["hourly"].get("precipitation_probability") or []
        precip = data["hourly"].get("precipitation") or []
        wp = [p for p, m in zip(probs, in_window) if m and p is not None]
        pr = [p for p, m in zip(precip, in_window) if m and p is not None]
        if not wp and not pr:
            return None  # window beyond forecast horizon
        rain_prob = max(wp) / 100.0 if wp else None
        precip_mm = float(sum(pr)) if pr else None
        wet = (rain_prob is not None and rain_prob >= WET_PROB_THRESHOLD) or (
            rain_prob is None and precip_mm is not None and precip_mm >= 0.5
        )
        out = {
            "rain_prob": rain_prob,
            "precip_mm": precip_mm,
            "is_wet": 1.0 if wet else 0.0,
            "window_start_utc": start.isoformat(),
            "source": "open-meteo",
        }
    except Exception:
        return None  # any failure: caller falls back to is_wet = NaN

    _cache[key] = (time.time(), out)
    return out
