import { useState, useSyncExternalStore } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../lib/api'
import { driverPhotoUrl } from '../lib/driverPhoto'
import { num, pct, pos, teamHex } from '../lib/format'
import { chart, label, status } from '../lib/theme'
import { useFetch } from '../lib/useFetch'
import { Card, ErrorCard, Skeleton, StatTile } from '../components/ui'

/* Phone-width flag for chart tick density/margins (ResponsiveContainer handles
   the size itself; this only thins labels so they never collide or clip). */
const NARROW_QUERY = '(max-width: 639px)'
function useNarrow() {
  return useSyncExternalStore(
    (cb) => {
      const mq = window.matchMedia(NARROW_QUERY)
      mq.addEventListener('change', cb)
      return () => mq.removeEventListener('change', cb)
    },
    () => window.matchMedia(NARROW_QUERY).matches,
    () => false,
  )
}

function SectionHead({ title, sub }: { title: string; sub?: string }) {
  return (
    <div className="mb-4 sm:mb-5">
      <h2 className={`${label} text-ink`}>{title}</h2>
      {sub && <p className="mt-1.5 text-xs text-ink3">{sub}</p>}
    </div>
  )
}

const th = `${label} py-2.5 pr-4 font-medium whitespace-nowrap text-ink3 last:pr-0`

export default function DriverDetail() {
  const { code = '' } = useParams()
  const { data, loading, error, reload } = useFetch(
    () => api.driverStats(code),
    [code],
  )
  const narrow = useNarrow()
  // remembers which driver's photo failed, so navigating to another driver retries
  const [badPhoto, setBadPhoto] = useState<string | null>(null)
  const photoOk = badPhoto !== code

  if (loading)
    return (
      <div className="space-y-6">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-56 sm:h-72" />
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-28" />
          ))}
        </div>
        <Skeleton className="h-72" />
      </div>
    )
  if (error || !data) return <ErrorCard message={error ?? 'No data'} onRetry={reload} />

  const color = teamHex(data.team_color)
  const chartData = data.recent_results.map((r) => ({
    name: `${r.event_name.replace(' Grand Prix', '')} '${String(r.season).slice(2)}`,
    finish: r.finish_pos,
    quali: r.quali_pos,
    dnf: r.dnf,
  }))
  const parts = data.driver_name.trim().split(/\s+/)
  const first = parts.length > 1 ? parts.slice(0, -1).join(' ') : ''
  const last = parts[parts.length - 1]
  // roughly 4 labels on a phone, every label (or close) on desktop
  const xInterval = narrow ? Math.max(0, Math.ceil(chartData.length / 4) - 1) : 'preserveStartEnd'

  return (
    <div className="space-y-8 sm:space-y-10">
      <Link
        to="/drivers"
        className={`${label} inline-flex min-h-11 items-center gap-2 text-ink3 transition-colors hover:text-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cream`}
      >
        <span aria-hidden>←</span> All drivers
      </Link>

      {/* Hero: name in Anton, photo on the team-colour baseline */}
      <section className="relative -mt-4 border-b border-line">
        <div className="grid items-end gap-4 sm:grid-cols-[minmax(0,1fr)_auto] sm:gap-8">
          <div className="@container relative order-2 min-w-0 pb-6 sm:order-1 sm:pb-8">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
              <span className={`${label} text-ink`}>#{data.driver_number}</span>
              <span className={`${label} flex min-w-0 items-center gap-2 text-ink2`}>
                <span aria-hidden className="inline-block h-2 w-5 shrink-0" style={{ background: color }} />
                <span className="truncate">{data.team}</span>
              </span>
            </div>
            <h1 className="mt-5 font-display leading-[0.88] tracking-[-0.01em] text-ink uppercase">
              {first && (
                <span className="mb-[0.3em] block text-[clamp(1.75rem,7vw,3.25rem)] text-ink2">{first}</span>
              )}
              {/* surname sized from its column width and letter count so long names
                  (HULKENBERG, ANTONELLI) stay on one line at every breakpoint */}
              <span
                className="block whitespace-nowrap"
                style={{ fontSize: `min(8.5rem, calc(175cqi / ${Math.max(last.length, 5)}))` }}
              >
                {last}
              </span>
            </h1>
            <div className="mt-6 grid grid-cols-2 gap-4 sm:mt-8 sm:flex sm:flex-wrap sm:items-end sm:gap-x-10">
              <div>
                <div className={`${label} text-ink3`}>Points this season</div>
                <div className="mt-2 font-display text-5xl leading-none text-accent tabular-nums sm:text-6xl">
                  {data.season.points}
                </div>
              </div>
              <div>
                <div className={`${label} text-ink3`}>Wins · podiums</div>
                <div className="mt-2 font-display text-5xl leading-none text-ink tabular-nums sm:text-6xl">
                  {data.season.wins}
                  <span className="text-ink3"> · </span>
                  {data.season.podiums}
                </div>
              </div>
            </div>
          </div>

          <div className="relative order-1 flex justify-center sm:order-2 sm:justify-end">
            <span
              aria-hidden
              className="pointer-events-none absolute right-0 bottom-0 font-display text-[9rem] leading-[0.8] text-ink/[0.06] select-none sm:text-[14rem]"
            >
              {data.driver_number}
            </span>
            {photoOk ? (
              <img
                src={driverPhotoUrl(data.driver_name, data.driver_code)}
                alt={data.driver_name}
                onError={() => setBadPhoto(code)}
                className="relative h-48 object-contain object-bottom sm:h-72 lg:h-80"
              />
            ) : (
              <span className="relative flex h-40 items-end pb-6 font-display text-6xl text-ink3 sm:h-72">
                {data.driver_code}
              </span>
            )}
          </div>
        </div>
        {/* team colour baseline */}
        <span aria-hidden className="absolute inset-x-0 -bottom-px h-0.5" style={{ background: color }} />
      </section>

      {/* Stat tiles */}
      <section>
        <SectionHead title="Career" sub="All races since 2022" />
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
          <StatTile label="Career wins" value={data.career.wins} sub="since 2022" />
          <StatTile label="Podiums" value={data.career.podiums} sub="since 2022" />
          <StatTile label="Poles" value={data.career.poles} sub="since 2022" />
          <StatTile
            label="Avg finish"
            value={num(data.career.avg_finish)}
            sub="all races since 2022"
          />
          <StatTile
            label="DNF rate"
            value={pct(data.career.dnf_rate)}
            sub="reliability + incidents"
          />
          <StatTile
            label="Overtakes"
            value={num(data.career.avg_overtakes)}
            sub="avg per race"
          />
        </div>
      </section>

      {/* Form chart */}
      <Card className="min-w-0 px-4 py-5 sm:px-6 sm:py-6">
        <SectionHead
          title={`Finishing position — last ${chartData.length} races`}
          sub="Higher is better (P1 at the top)"
        />
        <div className="h-56 sm:h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 8, right: narrow ? 12 : 20, left: 0, bottom: 0 }}
            >
              <CartesianGrid stroke={chart.grid} strokeWidth={1} vertical={false} />
              <XAxis
                dataKey="name"
                tick={{ ...chart.tick, fontSize: narrow ? 10 : 11 }}
                tickLine={false}
                axisLine={{ stroke: chart.axis }}
                interval={xInterval}
                minTickGap={narrow ? 4 : 12}
                tickMargin={8}
              />
              <YAxis
                reversed
                domain={[1, 20]}
                ticks={[1, 5, 10, 15, 20]}
                interval={0}
                tick={{ ...chart.tick, fontSize: narrow ? 10 : 11 }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v) => `P${v}`}
                width={narrow ? 30 : 36}
              />
              <Tooltip
                {...chart.tooltip}
                cursor={chart.cursor}
                formatter={(v) => [v == null ? 'DNF' : `P${v}`, 'Finish']}
              />
              <Line
                type="monotone"
                dataKey="finish"
                stroke={color}
                strokeWidth={2}
                strokeLinecap="round"
                dot={{ r: narrow ? 3 : 4, fill: color, stroke: chart.dotRing, strokeWidth: 2 }}
                activeDot={{ r: 6, stroke: chart.dotRing, strokeWidth: 2 }}
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Table twin of the chart */}
        <div className="-mx-4 mt-6 overflow-x-auto px-4 sm:mx-0 sm:px-0">
          <table className="w-full min-w-[28rem] sm:min-w-[34rem] text-sm tabular-nums">
            <thead>
              <tr className="border-b border-line text-left">
                <th className={th}>Race</th>
                <th className={`${th} text-right`}>Quali</th>
                <th className={`${th} text-right`}>Grid</th>
                <th className={`${th} text-right`}>Finish</th>
                <th className={`${th} text-right`}>Points</th>
                <th className={`${th} text-right`}>Conditions</th>
              </tr>
            </thead>
            <tbody>
              {[...data.recent_results].reverse().map((r) => (
                <tr
                  key={`${r.season}-${r.round}`}
                  className="border-b border-line last:border-0"
                >
                  <td className="py-2.5 pr-4 whitespace-nowrap text-ink2">
                    {narrow ? r.event_name.replace(' Grand Prix', '') : r.event_name} {r.season}
                  </td>
                  <td className="py-2.5 pr-4 text-right text-ink2">{pos(r.quali_pos)}</td>
                  <td className="py-2.5 pr-4 text-right text-ink2">{pos(r.grid_pos)}</td>
                  <td
                    className="py-2.5 pr-4 text-right font-semibold text-ink"
                    style={r.dnf ? { color: status.bad } : undefined}
                  >
                    {r.dnf ? 'DNF' : pos(r.finish_pos)}
                  </td>
                  <td className="py-2.5 pr-4 text-right text-ink2">{r.points}</td>
                  <td className="py-2.5 text-right text-ink3">
                    {r.wet_race ? 'Wet' : 'Dry'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Splits + track history */}
      <div className="grid gap-3 lg:grid-cols-3">
        <Card className="min-w-0 px-4 py-5 sm:px-6 sm:py-6">
          <SectionHead title="Performance profile" />
          <dl className="text-sm">
            {(
              [
                ['Avg finish in the wet', num(data.wet_avg_finish)],
                ['Avg finish in the dry', num(data.dry_avg_finish)],
                ['Avg positions gained', num(data.avg_positions_gained)],
                [
                  'Lap-time consistency',
                  data.consistency_ms == null
                    ? '—'
                    : `±${(data.consistency_ms / 1000).toFixed(2)}s`,
                ],
                ['Avg quali this season', num(data.season.avg_quali)],
              ] as const
            ).map(([k, v]) => (
              <div
                key={k}
                className="flex items-baseline justify-between gap-4 border-b border-line py-3 first:pt-0 last:border-0 last:pb-0"
              >
                <dt className="text-ink2">{k}</dt>
                <dd className="font-display text-xl leading-none text-ink tabular-nums">{v}</dd>
              </div>
            ))}
          </dl>
        </Card>

        <Card className="min-w-0 px-4 py-5 sm:px-6 sm:py-6 lg:col-span-2">
          <SectionHead title="Best tracks" sub="Average finish since 2022" />
          <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
            <table className="w-full min-w-[18rem] text-sm tabular-nums">
              <thead>
                <tr className="border-b border-line text-left">
                  <th className={th}>Track</th>
                  <th className={`${th} text-right`}>Races</th>
                  <th className={`${th} text-right`}>Avg finish</th>
                  <th className={`${th} text-right`}>Best</th>
                </tr>
              </thead>
              <tbody>
                {data.track_history.slice(0, 10).map((t) => (
                  <tr key={t.location} className="border-b border-line last:border-0">
                    <td className="py-2.5 pr-4 text-ink2">
                      <Link
                        to={`/tracks/${encodeURIComponent(t.location)}`}
                        className="underline decoration-line underline-offset-4 transition-colors hover:text-ink hover:decoration-accent"
                      >
                        {t.location}
                      </Link>
                    </td>
                    <td className="py-2.5 pr-4 text-right text-ink2">{t.races}</td>
                    <td className="py-2.5 pr-4 text-right text-ink">
                      {num(t.avg_finish)}
                    </td>
                    <td className="py-2.5 text-right font-semibold text-ink">
                      {pos(t.best_finish)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  )
}
