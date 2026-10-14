import type { CSSProperties, ReactNode } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../lib/api'
import { num, pct, pos, teamHex } from '../lib/format'
import { label } from '../lib/theme'
import { useFetch } from '../lib/useFetch'
import { ErrorCard, Skeleton, StatTile, TeamDot } from '../components/ui'

/* Section heading: tiny tracked label over a hairline, editorial style. */
function Section({ title, aside, children }: { title: string; aside?: string; children: ReactNode }) {
  return (
    <section className="min-w-0">
      <div className="flex items-baseline justify-between gap-4 border-b border-line pb-3">
        <h2 className={`${label} text-ink2`}>{title}</h2>
        {aside && <span className={`${label} text-ink3`}>{aside}</span>}
      </div>
      {children}
    </section>
  )
}

export default function TrackDetail() {
  const { location = '' } = useParams()
  const { data, loading, error, reload } = useFetch(
    () => api.trackStats(location),
    [location],
  )

  if (loading)
    return (
      <div className="space-y-4">
        <Skeleton className="h-6 w-28" />
        <Skeleton className="h-40" />
        <Skeleton className="h-28" />
        <Skeleton className="h-64" />
      </div>
    )
  if (error || !data) return <ErrorCard message={error ?? 'No data'} onRetry={reload} />

  const gridHold = data.grid_finish_correlation
  const gridHoldLabel =
    gridHold == null
      ? '—'
      : gridHold > 0.75
        ? 'Track position rules'
        : gridHold > 0.55
          ? 'Balanced'
          : 'Overtaking-friendly'

  const word = data.location.toUpperCase()

  return (
    <div className="space-y-10 sm:space-y-14">
      {/* hero: the location fills the column width (sized by character count
          against the container, so "SPA-FRANCORCHAMPS" still fits at 375px) */}
      <header className="border-b border-line pb-6 sm:pb-8">
        <Link
          to="/tracks"
          className={`${label} inline-flex min-h-11 items-center gap-2 text-ink3 transition-colors duration-200 hover:text-ink focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-cream`}
        >
          <span aria-hidden>←</span> All tracks
        </Link>
        <div className="mt-4 [container-type:inline-size]">
          <h1
            className="font-display leading-[0.88] tracking-[-0.01em] whitespace-nowrap text-ink"
            style={
              {
                fontSize: `min(11rem, calc(185cqw / ${Math.max(word.length, 6)}))`,
              } as CSSProperties
            }
          >
            {word}
          </h1>
        </div>
        <p className={`${label} mt-5 flex flex-wrap gap-x-3 gap-y-1 text-ink2`}>
          <span className="text-ink">{data.country}</span>
          <span aria-hidden className="text-ink3">·</span>
          <span>{data.races} editions since 2022</span>
        </p>
      </header>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
        <StatTile label="Avg overtakes" value={num(data.avg_overtakes, 0)} sub="per race" />
        <StatTile label="Safety car" value={pct(data.sc_rate)} sub="of editions" />
        <StatTile label="Virtual SC" value={pct(data.vsc_rate)} sub="of editions" />
        <StatTile label="Wet races" value={pct(data.wet_rate)} sub="of editions" />
        <StatTile label="Avg DNFs" value={num(data.avg_dnf)} sub="per race" />
        <StatTile
          label="Grid stickiness"
          value={gridHold == null ? '—' : num(gridHold, 2)}
          sub={gridHoldLabel}
        />
      </div>

      <div className="grid gap-10 lg:grid-cols-2 lg:gap-12">
        <Section title="Past winners" aside="Grid → win">
          <ol>
            {data.past_winners.map((w) => (
              <li
                key={w.season}
                className="flex items-center gap-4 border-b border-line py-3.5"
              >
                <span className="w-14 shrink-0 font-display text-2xl leading-none text-ink3">
                  {w.season}
                </span>
                <TeamDot color={teamHex(w.team_color)} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-semibold text-ink">
                    {w.driver_name}
                  </span>
                  <span className={`${label} mt-1 block truncate text-ink3`}>{w.team}</span>
                </span>
                <span className="shrink-0 text-right">
                  <span className={`${label} block text-ink3`}>From</span>
                  <span className="font-display text-xl leading-none text-ink">
                    {pos(w.grid_pos)}
                  </span>
                </span>
              </li>
            ))}
            {data.past_winners.length === 0 && (
              <li className="py-4 text-sm text-ink3">No editions ingested yet.</li>
            )}
          </ol>
        </Section>

        <Section title="Editions">
          <div className="overflow-x-auto">
            <table
              className="w-full min-w-[20rem] text-sm"
              style={{ fontVariantNumeric: 'tabular-nums' }}
            >
              <thead>
                <tr className={`${label} border-b border-line text-left whitespace-nowrap text-ink3 max-sm:tracking-[0.1em]`}>
                  <th className="py-3 pr-3 font-medium">Season</th>
                  <th className="py-3 pr-3 text-right font-medium">Overtakes</th>
                  <th className="py-3 pr-3 text-right font-medium">SC laps</th>
                  <th className="py-3 pr-3 text-right font-medium">DNFs</th>
                  <th className="py-3 text-right font-medium">
                    <span className="sm:hidden">Cond.</span>
                    <span className="hidden sm:inline">Conditions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.editions.map((e) => (
                  <tr key={e.season} className="border-b border-line">
                    <td className="py-3 pr-3 font-display text-xl leading-none text-ink">
                      {e.season}
                    </td>
                    <td className="py-3 pr-3 text-right text-ink">
                      {num(e.total_overtakes, 0)}
                    </td>
                    <td className="py-3 pr-3 text-right text-ink2">{e.sc_laps}</td>
                    <td className="py-3 pr-3 text-right text-ink2">{e.n_dnf}</td>
                    <td className="py-3 text-right">
                      {e.wet_race ? (
                        <span className={`${label} text-accent`}>Wet</span>
                      ) : (
                        <span className={`${label} text-ink3`}>Dry</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      </div>
    </div>
  )
}
