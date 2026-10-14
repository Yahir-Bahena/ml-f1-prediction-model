import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../lib/api'
import { etDateTime, num, pct } from '../lib/format'
import { label, status } from '../lib/theme'
import { useFetch } from '../lib/useFetch'
import ConfidenceCard from '../components/ConfidenceCard'
import Countdown from '../components/Countdown'
import PodiumHero from '../components/PodiumHero'
import PredictionTable from '../components/PredictionTable'
import WeatherChip from '../components/WeatherChip'
import { ErrorCard, Skeleton, StatTile } from '../components/ui'

/* Hairline-topped section heading: Anton title left, tiny tracked caption right. */
function SectionHead({ title, caption }: { title: string; caption?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-col gap-2 border-t border-line pt-5 sm:flex-row sm:items-baseline sm:justify-between sm:gap-6">
      <h2 className="shrink-0 font-display text-3xl leading-none text-ink uppercase sm:text-4xl">{title}</h2>
      {caption && <p className={`${label} leading-relaxed text-ink3 sm:text-right`}>{caption}</p>}
    </div>
  )
}

/* Flat rectangular tag used for the hero's status line. */
const tag = 'inline-flex max-w-full items-center gap-2 rounded-sm border px-2.5 py-1.5 text-[11px] leading-snug font-medium tracking-[0.12em] uppercase'

export default function Dashboard() {
  const pred = useFetch(api.predictions)
  const race = useFetch(api.nextRace)
  const metrics = useFetch(api.metrics)
  const track = useFetch(
    () =>
      pred.data
        ? api.trackStats(pred.data.event.location)
        : Promise.reject(new Error('waiting')),
    [pred.data?.event.location],
  )

  if (pred.loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-6 w-32" />
        <Skeleton className="h-20 w-3/4" />
        <Skeleton className="h-4 w-1/2" />
        <div className="grid gap-3 pt-6 lg:grid-cols-[1.65fr_1fr]">
          <Skeleton className="h-80" />
          <div className="grid gap-3">
            <Skeleton className="h-36" />
            <Skeleton className="h-36" />
          </div>
        </div>
        <Skeleton className="h-96" />
      </div>
    )
  }

  if (pred.error || !pred.data) {
    return (
      <ErrorCard
        message={pred.error ?? 'No prediction available yet.'}
        onRetry={pred.reload}
      />
    )
  }

  const { event, drivers, quali_available, grid_penalties, weather_forecast, confidence } =
    pred.data
  const penaltyCount = Object.keys(grid_penalties ?? {}).length

  return (
    <div className="space-y-12 sm:space-y-16">
      {/* Next race hero */}
      <header className="fade-up">
        <div className="flex flex-col gap-8 lg:flex-row lg:items-end lg:justify-between">
          <div className="min-w-0">
            <h1 className="font-display text-[clamp(2.75rem,10vw,6.5rem)] leading-[0.9] tracking-[-0.01em] break-words text-ink uppercase">
              {event.name}
            </h1>
            <p className={`${label} mt-4 leading-relaxed text-ink2`}>
              <span className="block sm:inline">{event.location}, {event.country}</span>
              <span aria-hidden className="mx-2 hidden text-ink3 sm:inline">·</span>
              <span className="block whitespace-nowrap sm:inline">{etDateTime(event.race_start_utc)}</span>
            </p>
            {race.data && (
              <p className="mt-1.5 text-xs text-ink3">{race.data.official_name}</p>
            )}
          </div>
          <div className="shrink-0">
            <div className={`${label} mb-2 text-ink3`}>Lights out in</div>
            <Countdown target={event.race_start_utc} />
          </div>
        </div>

        <div className="mt-6 flex flex-wrap items-center gap-2">
          <span
            className={`${tag} ${quali_available ? '' : 'border-line text-ink2'}`}
            style={
              quali_available
                ? { color: status.good, borderColor: `${status.good}55` }
                : undefined
            }
          >
            <span
              aria-hidden
              className="h-1.5 w-1.5 shrink-0 rounded-full bg-current"
            />
            {quali_available ? 'Includes qualifying results' : 'Pre-qualifying forecast'}
          </span>
          <WeatherChip forecast={weather_forecast} />
          {penaltyCount > 0 && (
            <span
              className={tag}
              style={{ color: status.bad, borderColor: `${status.bad}55` }}
              title={Object.entries(grid_penalties)
                .map(([c, p]) => `${c} +${p}`)
                .join(', ')}
            >
              {penaltyCount} grid {penaltyCount === 1 ? 'penalty' : 'penalties'} applied
            </span>
          )}
        </div>
      </header>

      {/* Predicted podium */}
      <section>
        <SectionHead
          title="Predicted podium"
          caption="Ranked by projected finish · win % breaks ties"
        />
        <PodiumHero top3={drivers.slice(0, 3)} />
      </section>

      {/* How sure is the model, and why */}
      {confidence && (
        <section>
          <SectionHead title="How sure we are" />
          <ConfidenceCard confidence={confidence} />
        </section>
      )}

      {/* Track + model stat tiles */}
      <section>
        <SectionHead title="By the numbers" caption={`${event.location} history · model backtest`} />
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-5 [&>*:last-child]:col-span-2 lg:[&>*:last-child]:col-span-1">
          <StatTile
            label="Track overtakes"
            value={track.data ? num(track.data.avg_overtakes, 0) : '—'}
            sub="avg per race here"
          />
          <StatTile
            label="Safety car"
            value={track.data ? pct(track.data.sc_rate) : '—'}
            sub="of past editions"
          />
          <StatTile
            label="Wet race odds"
            value={track.data ? pct(track.data.wet_rate) : '—'}
            sub="historically at this track"
          />
          <StatTile
            label="Winner hit rate"
            value={metrics.data ? pct(metrics.data.model.winner_hit_rate) : '—'}
            sub={
              metrics.data
                ? `backtested on ${metrics.data.model.races} races`
                : 'model accuracy'
            }
          />
          <StatTile
            label="Avg position error"
            value={metrics.data ? num(metrics.data.model.position_mae) : '—'}
            sub={
              metrics.data
                ? `vs ${num(metrics.data.baseline_grid.position_mae)} for grid-order`
                : 'model accuracy'
            }
          />
        </div>
      </section>

      {/* Full grid */}
      <section>
        <SectionHead
          title="Full grid"
          caption="Ranked by projected finish (win % breaks ties) · tap a row for the why"
        />
        <PredictionTable drivers={drivers} qualiAvailable={quali_available} />
      </section>

      <p className={`${label} border-t border-line pt-6 text-center leading-relaxed text-ink3`}>
        Model trained on 2022–{event.season} data ·{' '}
        <Link
          to="/tracks"
          className="inline-block py-2 text-ink2 underline decoration-line underline-offset-4 transition-colors hover:text-ink hover:decoration-accent"
        >
          Explore track stats →
        </Link>
      </p>
    </div>
  )
}
