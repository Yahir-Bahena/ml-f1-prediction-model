# F1 Race Predictor

A local web app that predicts the next Formula 1 race using FastF1 data from 2022 to the current season. A LightGBM ranking model scores every driver from ~28 engineered features (form, qualifying pace, track history, overtaking profiles, consistency, reliability, wet-weather skill…), then 10,000 Monte Carlo race simulations turn those scores into win / podium / points probabilities for the full grid.

## Stack

- **Backend** — Python, FastAPI, [FastF1](https://docs.fastf1.dev), LightGBM, pandas (`backend/`)
- **Frontend** — React + Vite + Tailwind CSS + Recharts (`frontend/`)

## Setup

### 1. Backend

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# macOS: LightGBM needs OpenMP
brew install libomp
```

### 2. Ingest data (one-time, slow)

Downloads race + qualifying data for 2022 → current season into `backend/data/`. FastF1's external APIs allow ~500 calls/hour — the script sleeps and resumes automatically when rate-limited, so the full first run takes a few hours. It is fully resumable; re-running only fetches missing races.

```bash
cd backend
.venv/bin/python -m scripts.ingest
```

### 3. Build dataset + train the model

```bash
.venv/bin/python -m ml.dataset
.venv/bin/python -m ml.train
```

Training walk-forward backtests the model (train on the past, predict each next race) and prints winner hit-rate, podium overlap, and mean position error against a "finish where you start" grid baseline. Metrics are saved next to the model and shown in the UI.

### 4. Run the app

```bash
# terminal 1 — API on :8000
cd backend && .venv/bin/uvicorn app.main:app --port 8000

# terminal 2 — UI on :5173
cd frontend && npm install && npm run dev
```

Open http://localhost:5173.

## After each race weekend

```bash
cd backend && ./scripts/retrain.sh
```

One command: ingests newly completed races, rebuilds the dataset, retrains. The running API picks up the new model automatically. If qualifying for the upcoming race has already happened, predictions automatically include the real grid; otherwise they're a pre-qualifying forecast.

## How the prediction works

1. **Ingest** (`scripts/ingest.py`) — for every race since 2022, compact per-driver rows (finish, grid, quali gaps, median pace, lap-time consistency, tyre-degradation slope, overtakes, lap-1 gains, DNFs, weather) plus per-race track stats (total overtakes, safety-car laps, wet).
2. **Features** (`ml/features.py`) — for each driver-race, ~28 features computed **only from earlier races** (no leakage): rolling form, season standings, track-specific history, team strength, overtaking records, reliability, wet-vs-dry deltas, plus that race's qualifying.
3. **Model** (`ml/train.py`) — LightGBM lambdarank ranker (one group per race) whose within-race ordering is blended with the qualifying-grid prior; NaN-native, so missing history (rookies, new tracks, no quali yet) is handled without imputation.
4. **Simulation** (`ml/simulate.py`) — each driver's score + race-day noise (calibrated from backtest residuals) + DNF sampling, 10,000 times → win/podium/points probabilities and expected-finish ranges.
5. **Explanations** — SHAP contributions per driver power the "why the model ranks them here" panel.

## API

| Endpoint | Description |
|---|---|
| `GET /api/predictions/next` | Full-grid prediction with probabilities + per-driver "why" |
| `GET /api/next-race` | Next event, sessions, countdown target |
| `GET /api/model/metrics` | Backtest accuracy + feature importance |
| `GET /api/drivers` · `/api/drivers/{code}/stats` | Lineup and deep driver stats |
| `GET /api/tracks` · `/api/tracks/{location}/stats` | Circuit character: overtakes, SC rate, winners |
| `GET /api/standings` | Driver + constructor standings |
| `GET /api/schedule?season=` | Season calendar with ingestion status |
