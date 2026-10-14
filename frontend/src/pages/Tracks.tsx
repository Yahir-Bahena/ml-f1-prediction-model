import { Link } from 'react-router-dom'
import { api } from '../lib/api'
import { num, pct } from '../lib/format'
import { label } from '../lib/theme'
import { useFetch } from '../lib/useFetch'
import { ErrorCard, PageTitle, Skeleton } from '../components/ui'

/* Editorial index: one hairline row per circuit, location set in Anton with the
   Grand Prix + country as tracked metadata. On md+ the numbers sit in aligned
   columns under a tiny header; on phones they fold into a 4-up stat strip. */

const COLS = 'lg:grid lg:grid-cols-[2.5rem_minmax(0,1fr)_repeat(5,6.25rem)_1.5rem] lg:items-center lg:gap-x-4'

export default function Tracks() {
  const { data, loading, error, reload } = useFetch(api.tracks)

  if (loading)
    return (
      <div className="space-y-3">
        <Skeleton className="h-28" />
        <Skeleton className="h-96" />
      </div>
    )
  if (error || !data) return <ErrorCard message={error ?? 'No data'} onRetry={reload} />

  return (
    <div>
      <PageTitle title="Tracks" />

      <div className="-mt-4 lg:mt-0" style={{ fontVariantNumeric: 'tabular-nums' }}>
        {/* column header (desktop only) */}
        <div className={`${label} hidden border-b border-line py-3 whitespace-nowrap text-ink3 ${COLS}`}>
          <span>No.</span>
          <span>Circuit</span>
          <span className="text-right">Races</span>
          <span className="text-right">Overtakes</span>
          <span className="text-right">Safety car</span>
          <span className="text-right">Wet</span>
          <span className="text-right">DNFs</span>
          <span />
        </div>

        <ol>
          {data.map((t, i) => (
            <li key={t.location} className="border-b border-line">
              <Link
                to={`/tracks/${encodeURIComponent(t.location)}`}
                className={`group block py-5 transition-colors duration-200 hover:bg-card focus-visible:bg-card focus-visible:outline-1 focus-visible:-outline-offset-1 focus-visible:outline-cream lg:py-4 ${COLS}`}
              >
                <span className={`${label} hidden text-ink3 lg:block`}>
                  {String(i + 1).padStart(2, '0')}
                </span>

                <span className="flex min-w-0 items-start gap-3">
                  <span className={`${label} w-6 shrink-0 pt-2 text-ink3 lg:hidden`}>
                    {String(i + 1).padStart(2, '0')}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block font-display text-[clamp(1.75rem,7vw,2.75rem)] leading-[0.95] break-words text-ink uppercase transition-colors duration-200 group-hover:text-accent">
                      {t.location}
                    </span>
                    <span className={`${label} mt-2 block text-ink3 lg:truncate`}>
                      {t.event_name} · {t.country}
                    </span>
                  </span>
                  <span
                    aria-hidden
                    className="shrink-0 pt-2 text-ink3 transition-transform duration-200 group-hover:translate-x-0.5 group-hover:text-accent lg:hidden"
                  >
                    →
                  </span>
                </span>

                {/* phones: compact stat strip */}
                <dl className="mt-4 ml-9 grid max-w-md grid-cols-[1.4fr_1fr_1fr_1fr] gap-x-3 lg:hidden">
                  <MobileStat k="Overtakes" v={num(t.avg_overtakes, 0)} />
                  <MobileStat k="SC" v={pct(t.sc_rate)} />
                  <MobileStat k="Wet" v={pct(t.wet_rate)} />
                  <MobileStat k="DNFs" v={num(t.avg_dnf)} />
                </dl>

                {/* desktop: aligned columns */}
                <span className="hidden text-right text-sm text-ink2 lg:block">{t.races}</span>
                <span className="hidden text-right font-display text-2xl leading-none text-ink lg:block">
                  {num(t.avg_overtakes, 0)}
                </span>
                <span className="hidden text-right text-sm text-ink2 lg:block">{pct(t.sc_rate)}</span>
                <span className="hidden text-right text-sm text-ink2 lg:block">{pct(t.wet_rate)}</span>
                <span className="hidden text-right text-sm text-ink2 lg:block">{num(t.avg_dnf)}</span>
                <span
                  aria-hidden
                  className="hidden text-right text-ink3 transition-transform duration-200 group-hover:translate-x-0.5 group-hover:text-accent lg:block"
                >
                  →
                </span>
              </Link>
            </li>
          ))}
        </ol>
      </div>
    </div>
  )
}

function MobileStat({ k, v }: { k: string; v: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-[10px] font-medium tracking-[0.16em] text-ink3 uppercase">{k}</dt>
      <dd className="mt-1 text-sm text-ink">{v}</dd>
    </div>
  )
}
