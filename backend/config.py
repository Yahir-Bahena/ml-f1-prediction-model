"""Shared paths and constants for the F1 predictor backend."""
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"          # fastf1 on-disk cache
PROCESSED_DIR = DATA_DIR / "processed"  # parquet tables built by ingest
MODELS_DIR = DATA_DIR / "models"        # trained model + metrics

RESULTS_PARQUET = PROCESSED_DIR / "race_results.parquet"
RACES_PARQUET = PROCESSED_DIR / "race_meta.parquet"

SEASONS = list(range(2022, 2027))
CURRENT_SEASON = 2026

# Points-scoring positions (top 10)
POINTS_POSITIONS = 10

# ---------------------------------------------------------------------------
# Canonical track labels.
# The fastf1 schedule renames locations/countries between seasons, which
# splits track history (e.g. Monaco vs Monte Carlo). Aliases are rewritten to
# one canonical value everywhere data enters the system; any location NOT in
# these maps passes through untouched, so new circuits are unaffected.
# ---------------------------------------------------------------------------
CANONICAL_LOCATION = {
    "Monte Carlo": "Monaco",   # 2026 rename; 2022-2025 use "Monaco"
    "Miami Gardens": "Miami",  # 2025+ rename; 2022-2024 use "Miami"
}

# canonical country per (canonical) location; cosmetic renames in the schedule
CANONICAL_COUNTRY = {
    "Silverstone": "United Kingdom",       # was "Great Britain" 2022-23
    "Yas Island": "United Arab Emirates",  # was "Abu Dhabi" 2022-23
}


def normalize_location(location: str) -> str:
    return CANONICAL_LOCATION.get(location, location)


def normalize_country(location: str, country: str) -> str:
    """Canonical country for a (already-normalized) location; pass-through otherwise."""
    return CANONICAL_COUNTRY.get(location, country)


def ensure_dirs() -> None:
    for d in (CACHE_DIR, PROCESSED_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_fastf1() -> None:
    """Enable the fastf1 cache and quiet its logging."""
    import fastf1

    ensure_dirs()
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.set_log_level("WARNING")
