import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { DriverSummary } from '../lib/api'
import { api } from '../lib/api'
import { driverPhotoUrl } from '../lib/driverPhoto'
import { num, teamHex } from '../lib/format'
import { label } from '../lib/theme'
import { useFetch } from '../lib/useFetch'
import { ErrorCard, PageTitle, Skeleton } from '../components/ui'

/* "Max Verstappen" → ["Max", "Verstappen"]; the surname carries the Anton. */
function splitName(name: string): [string, string] {
  const parts = name.trim().split(/\s+/)
  if (parts.length < 2) return ['', name]
  return [parts.slice(0, -1).join(' '), parts[parts.length - 1]]
}

function DriverCard({ d, delay }: { d: DriverSummary; delay: number }) {
  const [photoOk, setPhotoOk] = useState(true)
  const color = teamHex(d.team_color)
  const [first, last] = splitName(d.driver_name)
  return (
    <Link
      to={`/drivers/${d.driver_code}`}
      className="group block min-w-0 rounded-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cream"
    >
      <article
        className="rise relative flex h-full flex-col overflow-hidden rounded-sm border border-line bg-card transition-colors duration-200 group-hover:border-ink3"
        style={{ animationDelay: `${delay}ms` }}
      >
        {/* team colour as a thin top rule */}
        <span aria-hidden className="absolute inset-x-0 top-0 h-0.5" style={{ background: color }} />

        {/* photo stage with the race number as a watermark */}
        <div className="relative flex h-32 items-end justify-center overflow-hidden sm:h-44">
          <span
            aria-hidden
            className="pointer-events-none absolute -top-1 right-2 font-display text-[4.5rem] leading-none text-ink/[0.06] select-none sm:right-3 sm:text-[7rem]"
          >
            {d.driver_number}
          </span>
          {photoOk ? (
            <img
              src={driverPhotoUrl(d.driver_name, d.driver_code)}
              alt=""
              loading="lazy"
              onError={() => setPhotoOk(false)}
              className="relative h-28 object-contain object-bottom transition-transform duration-300 group-hover:-translate-y-1 motion-reduce:transition-none sm:h-40"
            />
          ) : (
            <span className="relative mb-5 font-display text-4xl text-ink3 sm:mb-8 sm:text-5xl">
              {d.driver_code}
            </span>
          )}
        </div>

        <div className="flex flex-1 flex-col border-t border-line px-3 pt-3 pb-3.5 sm:px-5 sm:pt-4 sm:pb-5">
          <div className="flex items-baseline justify-between gap-2">
            <span className={`${label} truncate text-ink3`}>{first || ' '}</span>
            <span className={`${label} shrink-0 text-ink3`}>{d.driver_code}</span>
          </div>
          <h3 className="mt-1 truncate font-display text-[1.6rem] leading-none text-ink uppercase sm:text-[2.25rem]">
            {last}
          </h3>

          <dl className="mt-auto grid grid-cols-2 gap-x-3 gap-y-2.5 border-t border-line pt-3 sm:mt-4 sm:grid-cols-4 sm:pt-4">
            {(
              [
                [d.season_points, 'Pts'],
                [d.wins, 'Wins'],
                [d.podiums, 'Pods'],
                [num(d.avg_finish), 'Avg fin'],
              ] as const
            ).map(([v, l]) => (
              <div key={l} className="min-w-0">
                <dt className="text-[10px] font-medium tracking-[0.16em] text-ink3 uppercase">{l}</dt>
                <dd className="mt-0.5 font-display text-xl leading-none text-ink tabular-nums sm:text-2xl">
                  {v}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </article>
    </Link>
  )
}

export default function Drivers() {
  const { data, loading, error, reload } = useFetch(api.drivers)

  if (loading)
    return (
      <div>
        <Skeleton className="mb-10 h-24 w-64" />
        <div className="space-y-10">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="space-y-4">
              <Skeleton className="h-8 w-56" />
              <div className="grid grid-cols-2 gap-3 sm:gap-4">
                <Skeleton className="h-64 sm:h-80" />
                <Skeleton className="h-64 sm:h-80" />
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  if (error || !data) return <ErrorCard message={error ?? 'No data'} onRetry={reload} />

  // group into teams, teams ordered by combined points, teammates by points
  const teams = new Map<string, DriverSummary[]>()
  for (const d of data) {
    teams.set(d.team, [...(teams.get(d.team) ?? []), d])
  }
  const ordered = [...teams.entries()]
    .map(([team, ds]) => ({
      team,
      color: teamHex(ds[0].team_color),
      total: ds.reduce((s, d) => s + d.season_points, 0),
      drivers: [...ds].sort((a, b) => b.season_points - a.season_points),
    }))
    .sort((a, b) => b.total - a.total)

  return (
    <div>
      <PageTitle title="Drivers" />
      <div className="space-y-12 sm:space-y-16">
        {ordered.map((t, ti) => (
          <section key={t.team} aria-labelledby={`team-${ti}`}>
            <header
              className="rise mb-4 flex items-end gap-3 border-b border-line pb-3 sm:mb-5 sm:gap-5"
              style={{ animationDelay: `${ti * 90}ms` }}
            >
              <span className="font-display text-3xl leading-none text-ink3 tabular-nums sm:text-5xl">
                {String(ti + 1).padStart(2, '0')}
              </span>
              <div className="min-w-0 flex-1">
                <div className={`${label} flex items-center gap-2 text-ink3`}>
                  <span aria-hidden className="inline-block h-2 w-5 shrink-0" style={{ background: t.color }} />
                  <span className="truncate">
                    <span className="sr-only">P{ti + 1} in constructors · </span>
                    {t.drivers.map((d) => d.driver_code).join(' · ')}
                  </span>
                </div>
                <h2
                  id={`team-${ti}`}
                  className="mt-1.5 truncate font-display text-2xl leading-none text-ink uppercase sm:text-4xl"
                >
                  {t.team}
                </h2>
              </div>
              <div className="shrink-0 text-right">
                <div className="font-display text-2xl leading-none text-ink tabular-nums sm:text-4xl">
                  {t.total}
                </div>
                <div className={`${label} mt-1.5 text-ink3`}>Pts</div>
              </div>
            </header>
            <div className="grid grid-cols-2 gap-3 sm:gap-4">
              {t.drivers.map((d, di) => (
                <DriverCard key={d.driver_code} d={d} delay={ti * 90 + di * 60 + 60} />
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  )
}
