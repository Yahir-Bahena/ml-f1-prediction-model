import { Link } from 'react-router-dom'
import { api } from '../lib/api'
import { teamHex } from '../lib/format'
import { label } from '../lib/theme'
import { useFetch } from '../lib/useFetch'
import { ErrorCard, PageTitle, ProbBar, Skeleton, TeamDot } from '../components/ui'

/* Editorial standings: Anton position + points, tiny tracked headers, hairline
   rows. Each row is a CSS grid; the wins/podiums columns only appear from sm up,
   so at 375px a row is just position · name (+ team-coloured bar) · points. */

const DRV_COLS =
  'grid grid-cols-[2.25rem_minmax(0,1fr)_3.75rem] items-center gap-x-3 sm:grid-cols-[2.75rem_minmax(0,1fr)_2.75rem_2.75rem_4.5rem] sm:gap-x-4'
const TEAM_COLS =
  'grid grid-cols-[2.25rem_minmax(0,1fr)_3.75rem] items-center gap-x-3 sm:grid-cols-[2.75rem_minmax(0,1fr)_2.75rem_4.5rem] sm:gap-x-4'

function Heading({ title, count }: { title: string; count: number }) {
  return (
    <div className="mb-1 flex items-baseline justify-between gap-4">
      <h2 className="font-display text-3xl leading-none text-ink uppercase sm:text-4xl">{title}</h2>
      <span className={`${label} text-ink3`}>{count} entries</span>
    </div>
  )
}

function Pos({ n }: { n: number }) {
  return (
    <span
      className={`font-display text-2xl leading-none sm:text-3xl ${n === 1 ? 'text-accent' : 'text-ink3'}`}
    >
      {n}
    </span>
  )
}

export default function Standings() {
  const { data, loading, error, reload } = useFetch(api.standings)

  if (loading)
    return (
      <div className="space-y-6">
        <Skeleton className="h-28" />
        <div className="grid gap-6 lg:grid-cols-2">
          <Skeleton className="h-96" />
          <Skeleton className="h-96" />
        </div>
      </div>
    )
  if (error || !data) return <ErrorCard message={error ?? 'No data'} onRetry={reload} />

  const maxDrv = Math.max(...data.drivers.map((d) => d.points), 1)
  const maxTeam = Math.max(...data.teams.map((t) => t.points), 1)

  return (
    <div>
      <PageTitle title={`${data.season} Standings`} />
      <div
        className="grid gap-12 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)] lg:gap-14"
        style={{ fontVariantNumeric: 'tabular-nums' }}
      >
        {/* drivers */}
        <section className="min-w-0">
          <Heading title="Drivers" count={data.drivers.length} />
          <div className={`${label} ${DRV_COLS} border-b border-line py-3 text-ink3`}>
            <span>Pos</span>
            <span>Driver</span>
            <span className="hidden text-right sm:block">Wins</span>
            <span className="hidden text-right sm:block">Pod.</span>
            <span className="text-right">Pts</span>
          </div>
          <ol>
            {data.drivers.map((d, i) => (
              <li key={d.driver_code} className={`${DRV_COLS} border-b border-line py-3`}>
                <Pos n={i + 1} />
                <div className="min-w-0">
                  <div className="flex min-w-0 items-center gap-2.5">
                    <TeamDot color={teamHex(d.team_color)} />
                    <Link
                      to={`/drivers/${d.driver_code}`}
                      className="shrink-0 text-sm font-semibold tracking-[0.08em] text-ink transition-colors duration-200 hover:text-accent focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-cream"
                    >
                      {d.driver_code}
                    </Link>
                    <span className="min-w-0 truncate text-sm text-ink2">{d.driver_name}</span>
                  </div>
                  <ProbBar
                    value={d.points / maxDrv}
                    color={teamHex(d.team_color)}
                    className="mt-2"
                  />
                </div>
                <span className="hidden text-right text-sm text-ink2 sm:block">{d.wins}</span>
                <span className="hidden text-right text-sm text-ink2 sm:block">{d.podiums}</span>
                <span className="text-right font-display text-2xl leading-none text-ink">
                  {d.points}
                </span>
              </li>
            ))}
          </ol>
        </section>

        {/* constructors */}
        <section className="min-w-0 lg:self-start">
          <Heading title="Constructors" count={data.teams.length} />
          <div className={`${label} ${TEAM_COLS} border-b border-line py-3 text-ink3`}>
            <span>Pos</span>
            <span>Team</span>
            <span className="hidden text-right sm:block">Wins</span>
            <span className="text-right">Pts</span>
          </div>
          <ol>
            {data.teams.map((t, i) => (
              <li key={t.team} className={`${TEAM_COLS} border-b border-line py-3`}>
                <Pos n={i + 1} />
                <div className="min-w-0">
                  <div className="flex min-w-0 items-center gap-2.5">
                    <TeamDot color={teamHex(t.team_color)} />
                    <span className="min-w-0 truncate text-sm font-semibold text-ink">{t.team}</span>
                  </div>
                  <ProbBar
                    value={t.points / maxTeam}
                    color={teamHex(t.team_color)}
                    className="mt-2"
                  />
                </div>
                <span className="hidden text-right text-sm text-ink2 sm:block">{t.wins}</span>
                <span className="text-right font-display text-2xl leading-none text-ink">
                  {t.points}
                </span>
              </li>
            ))}
          </ol>
        </section>
      </div>
    </div>
  )
}
