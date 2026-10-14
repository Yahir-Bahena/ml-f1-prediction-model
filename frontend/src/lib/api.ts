const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(BASE + path)
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return res.json()
}

export interface SessionInfo {
  name: string
  start_utc: string
}

export interface NextRace {
  season: number
  round: number
  name: string
  official_name: string
  location: string
  country: string
  race_start_utc: string
  sessions: SessionInfo[]
}

export interface WhyItem {
  feature: string
  label: string
  impact: number
  value: string | null
}

export interface DriverPrediction {
  predicted_rank: number
  driver_code: string
  driver_name: string
  driver_number: string
  team: string
  team_color: string
  quali_pos: number | null
  start_pos: number | null
  grid_penalty: number
  score: number
  win_prob: number
  podium_prob: number
  points_prob: number
  dnf_prob: number
  mean_pos: number
  p10_pos: number
  p90_pos: number
  why: WhyItem[]
  form_finish_5: number | null
  track_finish_avg: number | null
}

export interface WeatherForecast {
  rain_prob: number | null
  precip_mm: number | null
  is_wet: number
  window_start_utc: string
  source: string
}

export interface ConfidenceReason {
  kind: 'good' | 'warn'
  text: string
}

export interface Confidence {
  score: number
  level: 'high' | 'medium' | 'low'
  reasons: ConfidenceReason[]
}

export interface Prediction {
  event: {
    season: number
    round: number
    name: string
    official_name: string
    location: string
    country: string
    race_start_utc: string
  }
  quali_available: boolean
  grid_penalties: Record<string, number>
  weather_forecast: WeatherForecast | null
  sigma: number
  confidence: Confidence | null
  drivers: DriverPrediction[]
}

export interface ModelMetrics {
  trained_at: string
  n_rows: number
  n_races: number
  sigma: number
  model: BacktestMetrics
  baseline_grid: BacktestMetrics
  feature_importance: [string, number][]
}

export interface BacktestMetrics {
  races: number
  winner_hit_rate: number
  podium_overlap_avg: number
  position_mae: number
  spearman_avg: number
}

export interface DriverSummary {
  driver_code: string
  driver_name: string
  driver_number: string
  team: string
  team_color: string
  season_points: number
  wins: number
  podiums: number
  avg_finish: number | null
  avg_quali: number | null
}

export interface RaceResultRow {
  season: number
  round: number
  event_name: string
  location: string
  grid_pos: number | null
  quali_pos: number | null
  finish_pos: number | null
  points: number
  dnf: boolean
  wet_race: boolean
}

export interface DriverStats {
  driver_code: string
  driver_name: string
  driver_number: string
  team: string
  team_color: string
  career: {
    races: number
    wins: number
    podiums: number
    poles: number
    dnf_rate: number
    avg_finish: number
    avg_overtakes: number | null
  }
  season: {
    points: number
    wins: number
    podiums: number
    avg_finish: number | null
    avg_quali: number | null
  }
  consistency_ms: number | null
  avg_positions_gained: number | null
  wet_avg_finish: number | null
  dry_avg_finish: number | null
  recent_results: RaceResultRow[]
  track_history: {
    location: string
    races: number
    avg_finish: number
    best_finish: number
    avg_gain: number | null
  }[]
}

export interface TrackSummary {
  location: string
  country: string
  event_name: string
  races: number
  avg_overtakes: number | null
  sc_rate: number
  wet_rate: number
  avg_dnf: number
  last_season: number
}

export interface TrackStats {
  location: string
  country: string
  event_name: string
  races: number
  avg_overtakes: number | null
  sc_rate: number
  vsc_rate: number
  wet_rate: number
  avg_dnf: number
  avg_track_temp: number | null
  grid_finish_correlation: number | null
  past_winners: {
    season: number
    driver_name: string
    team: string
    team_color: string
    grid_pos: number | null
  }[]
  editions: {
    season: number
    round: number
    wet_race: boolean
    sc_laps: number
    total_overtakes: number | null
    n_dnf: number
  }[]
}

export interface Standings {
  season: number
  drivers: {
    driver_code: string
    driver_name: string
    team: string
    team_color: string
    points: number
    wins: number
    podiums: number
  }[]
  teams: {
    team: string
    team_color: string
    points: number
    wins: number
  }[]
}

export interface ScheduleEvent {
  round: number
  name: string
  location: string
  country: string
  race_start_utc: string | null
  completed: boolean
}

export interface ResultMetrics {
  winner_hit: boolean
  podium_overlap: number
  mae: number | null
  spearman: number | null
}

export interface ResultSummary {
  season: number
  round: number
  event_name: string
  location: string
  country: string
  date: string | null
  wet_race: boolean
  predicted_winner: string
  actual_winner: string | null
  metrics: ResultMetrics
  n_highlights: number
}

export interface ResultDriver {
  driver_code: string
  driver_name: string
  team: string
  team_color: string
  driver_number: string
  predicted_rank: number
  win_prob: number
  grid_pos: number | null
  actual_finish: number | null
  dnf: boolean
  status: string
  delta: number | null
}

export interface RaceHighlight {
  lap: number | null
  type: 'safety_car' | 'vsc' | 'red_flag' | 'collision' | 'penalty'
  drivers: string[]
  text: string
}

export interface ResultDetail extends ResultSummary {
  drivers: ResultDriver[]
  highlights: RaceHighlight[]
}

export const api = {
  nextRace: () => get<NextRace>('/next-race'),
  results: () => get<ResultSummary[]>('/results'),
  resultDetail: (season: number, round: number) =>
    get<ResultDetail>(`/results/${season}/${round}`),
  predictions: () => get<Prediction>('/predictions/next'),
  metrics: () => get<ModelMetrics>('/model/metrics'),
  drivers: () => get<DriverSummary[]>('/drivers'),
  driverStats: (code: string) => get<DriverStats>(`/drivers/${code}/stats`),
  tracks: () => get<TrackSummary[]>('/tracks'),
  trackStats: (location: string) =>
    get<TrackStats>(`/tracks/${encodeURIComponent(location)}/stats`),
  standings: () => get<Standings>('/standings'),
  schedule: (season?: number) =>
    get<{ season: number; events: ScheduleEvent[] }>(
      `/schedule${season ? `?season=${season}` : ''}`,
    ),
}
