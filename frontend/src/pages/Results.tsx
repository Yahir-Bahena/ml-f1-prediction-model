import { Link } from 'react-router-dom'
import { api } from '../lib/api'
import { useFetch } from '../lib/useFetch'
import { ErrorCard, PageTitle, Skeleton } from '../components/ui'
import { label, status } from '../lib/theme'

/* Hit/miss marker: colour + glyph + word, so it never relies on colour alone. */
function HitMark({ hit }: { hit: boolean }) {
  return (
    <span
      className={`${label} inline-flex items-center gap-1.5 whitespace-nowrap tracking-[0.16em]`}
      style={{ color: hit ? status.good : status.bad }}
    >
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5" aria-hidden>
        {hit ? <path d="m3 8.5 3 3 7-7" /> : <path d="m4 4 8 8M12 4l-8 8" />}
      </svg>
      {hit ? 'Hit' : 'Miss'}
    </span>
  )
}

export default function Results() {
  const { data, loading, error, reload } = useFetch(api.results)

  if (loading)
    return (
      <div>
        <Skeleton className="mb-10 h-32" />
        <div className="space-y-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      </div>
    )
  if (error || !data) return <ErrorCard message={error ?? 'No data'} onRetry={reload} />

  const hits = data.filter((r) => r.metrics.winner_hit).length
  const pct = data.length ? Math.round((hits / data.length) * 100) : 0
  const podiumAvg = data.length
    ? data.reduce((s, r) => s + r.metrics.podium_overlap, 0) / data.length
    : 0

  return (
    <div>
      <PageTitle title="Track Record" />

      {data.length > 0 && (
        <section aria-label="Summary" className="grid grid-cols-2 gap-x-6 gap-y-8 border-b border-line pb-10 sm:grid-cols-3">
          <div className="col-span-2 sm:col-span-1">
            <p className={`${label} text-ink3`}>Winners called</p>
            <p className="mt-2 font-display text-[clamp(4rem,16vw,7rem)] leading-[0.9] text-ink">
              {pct}
              <span className="text-accent">%</span>
            </p>
            <div className="mt-4 h-1.5 w-full max-w-xs overflow-hidden bg-line">
              <div className="bar-fill h-full bg-rust" style={{ width: `${pct}%` }} />
            </div>
          </div>
          <div className="sm:self-end">
            <p className={`${label} text-ink3`}>Correct / called</p>
            <p className="mt-2 font-display text-5xl leading-none text-ink sm:text-6xl">
              {hits}
              <span className="text-ink3">/{data.length}</span>
            </p>
          </div>
          <div className="sm:self-end">
            <p className={`${label} text-ink3`}>Avg podium picks</p>
            <p className="mt-2 font-display text-5xl leading-none text-ink sm:text-6xl">
              {podiumAvg.toFixed(1)}
              <span className="text-ink3">/3</span>
            </p>
          </div>
        </section>
      )}

      {data.length === 0 ? (
        <p className="text-ink3">
          No completed races predicted yet. The first record appears after the
          next Grand Prix finishes.
        </p>
      ) : (
        <ol>
          {data.map((r, i) => {
            const hit = r.metrics.winner_hit
            const date = r.date ? new Date(r.date) : null
            return (
              <li key={`${r.season}-${r.round}`} className="rise border-b border-line" style={{ animationDelay: `${i * 50}ms` }}>
                <Link
                  to={`/results/${r.season}/${r.round}`}
                  className="group grid grid-cols-1 items-center sm:grid-cols-[1fr_auto] gap-x-4 gap-y-5 py-6 transition-colors duration-200 hover:bg-card focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cream sm:-mx-4 sm:px-4 lg:grid-cols-[5.5rem_minmax(0,1fr)_15rem_12.5rem] lg:gap-x-8"
                >
                  {/* round + date */}
                  <div className={`${label} flex items-center gap-3 text-ink3 sm:col-span-2 lg:col-span-1 lg:flex-col lg:items-start lg:gap-1`}>
                    <span className="text-accent">Rd {r.round}</span>
                    <span>
                      {date?.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                    </span>
                  </div>

                  {/* race */}
                  <div className="min-w-0 sm:col-span-2 lg:col-span-1">
                    <h2 className="font-display text-[clamp(1.75rem,7vw,2.75rem)] leading-[0.95] break-words text-ink uppercase transition-colors group-hover:text-accent2">
                      {r.location}
                    </h2>
                    <p className={`${label} mt-2 text-ink2`}>
                      {r.event_name}
                      {r.wet_race && <span className="text-ink3"> · Wet</span>}
                    </p>
                  </div>

                  {/* called vs actual */}
                  <div className="flex items-end gap-4">
                    <div>
                      <p className={`${label} text-ink3`}>Called</p>
                      <p className="mt-1 font-display text-3xl leading-none text-ink">{r.predicted_winner}</p>
                    </div>
                    <span aria-hidden className="pb-1 text-ink3">→</span>
                    <div>
                      <p className={`${label} text-ink3`}>Won</p>
                      <p
                        className="mt-1 font-display text-3xl leading-none"
                        style={{ color: hit ? status.good : undefined }}
                      >
                        {r.actual_winner ?? '—'}
                      </p>
                    </div>
                    <div className="pb-0.5 pl-1">
                      <HitMark hit={hit} />
                    </div>
                  </div>

                  {/* metrics */}
                  <dl className="flex items-end gap-6 sm:justify-end sm:gap-5 sm:text-right">
                    <div>
                      <dt className={`${label} text-ink3`}>Podium</dt>
                      <dd className="mt-1 text-sm font-semibold text-ink">{r.metrics.podium_overlap}/3</dd>
                    </div>
                    <div>
                      <dt className={`${label} text-ink3`}>Err</dt>
                      <dd className="mt-1 text-sm font-semibold text-ink">{r.metrics.mae?.toFixed(1) ?? '—'}</dd>
                    </div>
                    <div>
                      <dt className={`${label} text-ink3`}>Events</dt>
                      <dd className="mt-1 text-sm font-semibold text-ink">{r.n_highlights}</dd>
                    </div>
                  </dl>
                </Link>
              </li>
            )
          })}
        </ol>
      )}
    </div>
  )
}
