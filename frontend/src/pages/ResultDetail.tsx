import { Link, useParams } from 'react-router-dom'
import type { RaceHighlight, ResultDriver } from '../lib/api'
import { api } from '../lib/api'
import { teamHex } from '../lib/format'
import { useFetch } from '../lib/useFetch'
import { Card, ErrorCard, Skeleton, TeamDot } from '../components/ui'
import { label, status } from '../lib/theme'

/* accuracy of the actual finish relative to what was predicted. Colour plus a
   glyph so it reads without colour:
   good ✓ = exact hit
   bad  ↓ = model placed them HIGHER than they finished (over-predicted)
   warn ↑ = they finished HIGHER than the model predicted (out-performed) */
function accuracy(d: ResultDriver): { color: string | null; glyph: string; title: string } {
  if (d.actual_finish == null) return { color: null, glyph: '', title: 'Not classified' }
  const delta = d.actual_finish - d.predicted_rank
  if (delta === 0) return { color: status.good, glyph: '✓', title: 'Predicted correctly' }
  if (delta > 0) return { color: status.bad, glyph: '↓', title: 'Finished lower than predicted' }
  return { color: status.warn, glyph: '↑', title: 'Finished higher than predicted' }
}

function HighlightIcon({ type }: { type: RaceHighlight['type'] }) {
  const common = 'h-4 w-4 shrink-0'
  if (type === 'collision')
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke={status.bad} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className={common}>
        <path d="m13 2-3 7h5l-4 8 1-6H8l3-9M18 5l2-1M19 10l2 1M5 6 3 5M5 12l-2 1" />
      </svg>
    )
  if (type === 'penalty')
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke={status.warn} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className={common}>
        <circle cx="12" cy="12" r="9" /><path d="M12 8v5M12 16v.1" />
      </svg>
    )
  if (type === 'red_flag')
    return (
      <svg viewBox="0 0 24 24" fill={status.bad} stroke={status.bad} strokeWidth="1.5" strokeLinejoin="round" className={common}>
        <path d="M5 3v18M5 4h13l-2.5 4.5L18 13H5" />
      </svg>
    )
  // safety car / vsc
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke={status.warn} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className={common}>
      <path d="M5 16v-3l2-4h10l2 4v3M7 16v2M17 16v2M3 13h18" /><circle cx="8" cy="16" r="1.2" /><circle cx="16" cy="16" r="1.2" />
    </svg>
  )
}

/* movement DURING the race: grid start -> classified finish */
function RaceMovement({ d }: { d: ResultDriver }) {
  if (d.dnf)
    return (
      <span className="text-xs font-semibold" style={{ color: status.bad }}>
        DNF
      </span>
    )
  if (d.grid_pos == null || d.actual_finish == null)
    return <span className="text-xs text-ink3">—</span>
  const moved = d.grid_pos - d.actual_finish
  if (moved === 0) return <span className="text-xs text-ink3">held P{d.actual_finish}</span>
  return (
    <span className="text-xs font-semibold whitespace-nowrap" style={{ color: moved > 0 ? status.good : status.bad }}>
      {moved > 0 ? `▲ ${moved}` : `▼ ${-moved}`}
      <span className="ml-1 font-normal text-ink3">
        P{d.grid_pos}→P{d.actual_finish}
      </span>
    </span>
  )
}

function ComparisonChart({ drivers }: { drivers: ResultDriver[] }) {
  return (
    <div role="table" aria-label="Predicted order vs actual finish">
      {/* column headers */}
      <div role="row" className={`${label} flex items-center gap-2.5 border-b border-line px-1 pb-3 text-ink3 sm:gap-3 sm:px-2`}>
        <span role="columnheader" className="w-7 text-center">Pred</span>
        <span className="w-2.5" />
        <span role="columnheader" className="flex-1">Driver</span>
        <span role="columnheader" className="w-10 text-right">Win</span>
        <span role="columnheader" className="w-16 text-center sm:w-24">Actual</span>
        <span role="columnheader" className="hidden w-32 text-right sm:block">In the race</span>
      </div>
      <div>
        {drivers.map((d) => {
          const acc = accuracy(d)
          return (
            <div
              key={d.driver_code}
              role="row"
              className="flex items-center gap-2.5 border-b border-line/60 px-1 py-2.5 text-sm transition-colors last:border-b-0 hover:bg-card2/60 sm:gap-3 sm:px-2"
            >
              <span role="cell" className="w-7 text-center font-display text-xl leading-none text-ink">
                {d.predicted_rank}
              </span>
              <TeamDot color={teamHex(d.team_color)} />
              <span role="cell" className="min-w-0 flex-1">
                <span className="flex items-baseline gap-2">
                  <span className="font-semibold tracking-wide text-ink">{d.driver_code}</span>
                  <span className="truncate text-ink3">{d.driver_name}</span>
                </span>
                {/* on phones the race-movement column folds under the name */}
                <span className="mt-0.5 block sm:hidden">
                  <RaceMovement d={d} />
                </span>
              </span>
              <span role="cell" className="w-10 text-right text-xs text-ink2 tabular-nums">
                {(d.win_prob * 100).toFixed(0)}%
              </span>
              <span
                role="cell"
                className="w-16 rounded-sm border px-1 py-1 sm:w-24 text-center text-xs font-bold whitespace-nowrap tabular-nums"
                style={
                  acc.color
                    ? { color: acc.color, borderColor: `${acc.color}55`, background: `${acc.color}14` }
                    : { color: 'var(--color-ink3)', borderColor: 'var(--color-line)' }
                }
                title={acc.title}
              >
                <span className="sr-only">{acc.title}: </span>
                P{d.actual_finish ?? '—'}
                {acc.glyph && <span aria-hidden className="ml-1">{acc.glyph}</span>}
                {d.dnf && <span className="ml-0.5 hidden font-normal opacity-70 sm:inline">·DNF</span>}
              </span>
              <span role="cell" className="hidden w-32 text-right sm:block">
                <RaceMovement d={d} />
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function LegendItem({ color, glyph, text }: { color: string; glyph: string; text: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span
        aria-hidden
        className="inline-flex h-4 w-4 items-center justify-center rounded-sm border text-[10px] font-bold"
        style={{ color, borderColor: `${color}55`, background: `${color}14` }}
      >
        {glyph}
      </span>
      {text}
    </span>
  )
}

export default function ResultDetail() {
  const { season, round } = useParams()
  const { data, loading, error, reload } = useFetch(
    () => api.resultDetail(Number(season), Number(round)),
    [season, round],
  )

  if (loading)
    return (
      <div className="space-y-6">
        <Skeleton className="h-40" />
        <Skeleton className="h-96" />
      </div>
    )
  if (error || !data) return <ErrorCard message={error ?? 'Not found'} onRetry={reload} />

  const date = data.date ? new Date(data.date) : null
  const m = data.metrics
  const verdictColor = m.winner_hit ? status.good : status.bad

  return (
    <div className="space-y-10 sm:space-y-12">
      {/* hero */}
      <header>
        <Link
          to="/results"
          className={`${label} inline-flex min-h-11 items-center gap-2 text-ink3 transition-colors hover:text-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cream`}
        >
          <span aria-hidden>←</span> Track Record
        </Link>
        <h1 className="mt-4 font-display text-[clamp(2.5rem,11vw,6rem)] leading-[0.92] tracking-[-0.01em] break-words text-ink uppercase">
          {data.event_name}
        </h1>
        <div className="mt-5 flex flex-wrap items-center justify-between gap-x-6 gap-y-4">
          <p className={`${label} leading-relaxed text-ink2`}>
            {data.location}, {data.country}
            {date && ` · ${date.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' })}`}
            {data.wet_race && ' · Wet race'}
          </p>
          <span
            className={`${label} inline-flex items-center gap-2 rounded-full border px-4 py-1.5 tracking-[0.16em]`}
            style={{ color: verdictColor, borderColor: `${verdictColor}66` }}
          >
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5" aria-hidden>
              {m.winner_hit ? <path d="m3 8.5 3 3 7-7" /> : <path d="m4 4 8 8M12 4l-8 8" />}
            </svg>
            {m.winner_hit ? 'Winner called correctly' : 'Winner missed'}
          </span>
        </div>
      </header>

      {/* headline metrics */}
      <section aria-label="Race metrics" className="grid grid-cols-3 divide-x divide-line border-y border-line">
        {[
          [m.podium_overlap + '/3', 'Podium picks correct'],
          [m.mae?.toFixed(2) ?? '—', 'Avg position error'],
          [m.spearman?.toFixed(2) ?? '—', 'Rank correlation'],
        ].map(([v, text]) => (
          <div key={text as string} className="min-w-0 px-2 py-5 first:pl-0 sm:px-6 sm:py-6 sm:first:pl-0">
            <div className="font-display text-[clamp(2rem,9vw,4rem)] leading-none text-ink">{v}</div>
            <div className={`${label} mt-2 leading-snug tracking-[0.14em] text-ink3 sm:tracking-[0.2em]`}>{text}</div>
          </div>
        ))}
      </section>

      {/* predicted order with colour-coded actual finish + race movement */}
      <section>
        <div className="mb-4 flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
          <h2 className="font-display text-3xl leading-none text-ink uppercase sm:text-4xl">
            Predicted vs actual
          </h2>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px] text-ink2">
            <LegendItem color={status.good} glyph="✓" text="predicted correctly" />
            <LegendItem color={status.bad} glyph="↓" text="finished lower" />
            <LegendItem color={status.warn} glyph="↑" text="finished higher" />
          </div>
        </div>
        <Card className="px-2 py-3 sm:px-4 sm:py-4">
          <ComparisonChart drivers={data.drivers} />
        </Card>
      </section>

      {/* race control timeline */}
      {data.highlights.length > 0 && (
        <section>
          <h2 className="font-display text-3xl leading-none text-ink uppercase sm:text-4xl">
            What happened
          </h2>
          <p className="mt-3 mb-5 max-w-2xl text-sm text-ink3">
            Key race-control events. These are for context only — the model never
            sees them; its prediction is made before the race.
          </p>
          <ol className="border-t border-line">
            {data.highlights.map((h, i) => (
              <li key={i} className="flex items-start gap-4 border-b border-line py-4">
                <span className={`${label} w-14 shrink-0 pt-0.5 text-accent`}>
                  {h.lap != null ? `Lap ${h.lap}` : '—'}
                </span>
                <span className="mt-0.5"><HighlightIcon type={h.type} /></span>
                <div className="min-w-0 flex-1">
                  <div className="text-sm text-ink">{h.text}</div>
                  {h.drivers.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {h.drivers.map((c) => (
                        <span key={c} className="rounded-sm border border-line bg-card2 px-1.5 py-0.5 text-xs font-semibold text-ink2">
                          {c}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </li>
            ))}
          </ol>
        </section>
      )}
    </div>
  )
}
