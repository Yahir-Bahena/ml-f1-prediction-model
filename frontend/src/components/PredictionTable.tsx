import { Fragment, useEffect, useRef, useState } from 'react'
import type { DriverPrediction } from '../lib/api'
import { driverPhotoUrl } from '../lib/driverPhoto'
import { num, pct, pos, teamHex } from '../lib/format'
import { label as labelCls, status } from '../lib/theme'
import { Card, InfoTip, ProbBar, TeamDot } from './ui'

function StatRow({
  label,
  info,
  children,
}: {
  label: string
  info: string
  children: React.ReactNode
}) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-line/70 py-2 last:border-0">
      <span className="flex items-center gap-1 text-ink3">
        {label}
        <InfoTip text={info} />
      </span>
      <span className="text-right text-ink">{children}</span>
    </div>
  )
}

function DriverPhotoCard({ d }: { d: DriverPrediction }) {
  const [photoOk, setPhotoOk] = useState(true)
  const color = teamHex(d.team_color)
  return (
    <div className="relative flex w-full flex-row items-end overflow-hidden rounded-sm border border-line bg-card sm:w-44 sm:flex-col sm:items-center">
      <div aria-hidden className="absolute inset-x-0 top-0 h-[3px]" style={{ background: color }} />
      <span
        aria-hidden
        className="pointer-events-none absolute top-2 right-3 font-display text-[4.5rem] leading-none text-ink/[0.06] select-none sm:right-auto"
      >
        {d.driver_number}
      </span>
      {photoOk ? (
        <img
          src={driverPhotoUrl(d.driver_name, d.driver_code)}
          alt={d.driver_name}
          loading="lazy"
          onError={() => setPhotoOk(false)}
          className="relative mt-3 h-28 w-24 shrink-0 object-contain object-bottom sm:h-32 sm:w-auto"
        />
      ) : (
        <span className="relative mt-3 flex h-28 w-24 shrink-0 items-end justify-center pb-4 font-display text-3xl text-ink3 uppercase sm:h-32 sm:w-full">
          {d.driver_code}
        </span>
      )}
      <div className="relative min-w-0 flex-1 px-3 py-3 sm:w-full sm:flex-none sm:border-t sm:border-line sm:text-center">
        <div className="truncate font-display text-xl leading-none text-ink uppercase">{d.driver_name}</div>
        <div className={`${labelCls} mt-2 flex items-center gap-1.5 text-ink3 sm:justify-center`}>
          <span className="inline-block h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: color }} />
          <span className="truncate">{d.team}</span>
        </div>
      </div>
    </div>
  )
}

function WhyRow({ d }: { d: DriverPrediction }) {
  // spread of simulated finishes — deliberately NOT called
  // "confidence": that word belongs to the race-level win call
  const spread = d.p90_pos - d.p10_pos
  const [spreadLabel, spreadColor] =
    spread <= 5
      ? ['tight spread', status.good]
      : spread <= 9
        ? ['moderate spread', status.warn]
        : ['wide spread', status.bad]

  return (
    <div className="grid gap-5 border-t border-line bg-card2/40 px-4 py-5 sm:px-5 md:grid-cols-[auto_1fr_1fr]">
      <DriverPhotoCard d={d} />
      <div className="min-w-0">
        <div className={`${labelCls} mb-3 text-ink3`}>Why the model ranks {d.driver_code} here</div>
        <span
          className="mb-4 inline-block rounded-sm border px-2 py-1 text-[11px] font-medium tracking-[0.08em] uppercase"
          style={{ color: spreadColor, borderColor: `${spreadColor}66` }}
          title={`In 80% of the 10,000 simulations this driver finishes between P${d.p10_pos.toFixed(0)} and P${d.p90_pos.toFixed(0)}. This is about finishing-position precision — separate from the race-level win confidence.`}
        >
          Finish P{d.p10_pos.toFixed(0)}–P{d.p90_pos.toFixed(0)} · {spreadLabel}
        </span>
        <ul className="space-y-2">
          {d.why.length === 0 && (
            <li className="text-sm text-ink3">No single factor stands out.</li>
          )}
          {d.why.map((w) => (
            <li key={w.feature} className="flex items-start gap-2.5 text-sm leading-snug">
              <span
                className="inline-flex w-14 shrink-0 justify-center rounded-sm bg-card2 px-1.5 py-0.5 text-xs font-semibold"
                style={{
                  color: w.impact >= 0 ? status.good : status.bad,
                  fontVariantNumeric: 'tabular-nums',
                }}
              >
                {w.impact >= 0 ? '▲' : '▼'} {Math.abs(w.impact).toFixed(1)}
              </span>
              <span className="min-w-0 text-ink2">
                {w.label}
                {w.value != null && <span className="text-ink3"> ({w.value})</span>}
              </span>
            </li>
          ))}
        </ul>
        <div className="mt-3 text-xs text-ink3">
          ▲ pushes toward P1, ▼ pushes down the order (positions of impact).
        </div>
      </div>
      <div className="min-w-0 self-start text-sm">
        <StatRow
          label="Expected finish"
          info="The driver's average finishing position across 10,000 simulated runs of this race — the board is ranked by this. The range covers 80% of simulated outcomes. When two drivers' projected finishes are within ~0.25 of a position (effectively tied), the higher win % is ranked ahead."
        >
          {pos(d.mean_pos)}{' '}
          <span className="text-ink3">
            (range {pos(d.p10_pos)}–{pos(d.p90_pos)})
          </span>
        </StatRow>
        <StatRow
          label="Points probability"
          info="Chance of finishing in the top 10 and scoring championship points."
        >
          {pct(d.points_prob)}
        </StatRow>
        <StatRow
          label="DNF risk"
          info="Chance of not finishing the race, estimated from the driver's and the car's recent reliability record."
        >
          {pct(d.dnf_prob)}
        </StatRow>
        <StatRow
          label="Avg finish, last 5 races"
          info="Not points — race positions. The driver's average finishing position over their last 5 Grands Prix; 3.2 means they typically finished around 3rd. Lower is better."
        >
          {num(d.form_finish_5)}
        </StatRow>
        <StatRow
          label="Avg finish at this track"
          info="The driver's average finishing position in past races at this circuit specifically. Lower is better; blank if they've never raced here."
        >
          {num(d.track_finish_avg)}
        </StatRow>
      </div>
    </div>
  )
}

const PROJ_INFO =
  'Average finishing position across 10,000 simulations — lower is better, and the board is ranked by it. When drivers are within ~0.25 of a position (effectively tied), the higher win % ranks ahead, so a clear win favourite is never shown below a near-identical projected finish with lower win odds.'

/* Sticky first column: opaque background so scrolled cells pass beneath it,
   plus a hairline on its right edge. */
const STICKY = 'sticky left-0 z-10 bg-card shadow-[inset_-1px_0_0_var(--color-line)] sm:shadow-none'

export default function PredictionTable({
  drivers,
  qualiAvailable,
}: {
  drivers: DriverPrediction[]
  qualiAvailable: boolean
}) {
  const [open, setOpen] = useState<string | null>(null)
  const field = drivers.length

  // On phones the table scrolls sideways; the expanded "why" panel is pinned to
  // the visible width of the scroller (sticky + measured width) so it never
  // needs horizontal scrolling itself.
  const scroller = useRef<HTMLDivElement>(null)
  const [viewW, setViewW] = useState<number | null>(null)
  const [scrolls, setScrolls] = useState(false)
  useEffect(() => {
    const el = scroller.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      setViewW(el.clientWidth)
      setScrolls(el.scrollWidth > el.clientWidth + 1)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const toggle = (code: string, expanded: boolean) => setOpen(expanded ? null : code)

  return (
    <Card>
      {/* phones: the header ⓘ would be clipped by the scroller, so the
          explanation lives in this strip instead, with a swipe hint */}
      <div className={`${labelCls} flex items-center justify-between gap-3 border-b border-line px-4 py-2.5 text-ink3 md:hidden`}>
        <span className="flex items-center gap-1">
          Proj. finish
          <InfoTip text={PROJ_INFO} />
        </span>
        {scrolls && <span aria-hidden>Swipe for more →</span>}
      </div>
      <div ref={scroller} className="relative overflow-x-auto overscroll-x-contain md:overflow-visible">
        <table className="w-full min-w-[21rem] text-sm sm:min-w-[34rem]" style={{ fontVariantNumeric: 'tabular-nums' }}>
          <thead>
            <tr className={`${labelCls} border-b border-line text-left text-ink3`}>
              <th className={`${STICKY} py-3 pr-2 pl-4 font-medium sm:pl-5`}>
                <span className="inline-block w-7">#</span>Driver
              </th>
              {qualiAvailable && (
                <th className="px-2 py-3 text-right font-medium">Start</th>
              )}
              <th className="px-2 py-3 text-right font-medium whitespace-nowrap sm:px-4 sm:text-left sm:w-[26%]">
                <span className="inline-flex items-center gap-1">
                  Proj. finish
                  <span className="hidden md:inline-flex">
                    <InfoTip text={PROJ_INFO} />
                  </span>
                </span>
              </th>
              <th className="px-2 py-3 text-right font-medium">Win</th>
              <th className="px-2 py-3 text-right font-medium">Podium</th>
              <th className="px-2 py-3 text-right font-medium">Points</th>
              <th className="w-10 px-2 py-3 sm:pr-4">
                <span className="sr-only">Details</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {drivers.map((d) => {
              const color = teamHex(d.team_color)
              const expanded = open === d.driver_code
              const top3 = d.predicted_rank <= 3
              return (
                <Fragment key={d.driver_code}>
                  <tr
                    className={`group h-12 cursor-pointer border-b border-line/70 transition-colors last:border-0 hover:bg-card2/60 focus-visible:outline-1 focus-visible:-outline-offset-1 focus-visible:outline-cream ${expanded ? 'bg-card2/60' : ''}`}
                    onClick={() => toggle(d.driver_code, expanded)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        toggle(d.driver_code, expanded)
                      }
                    }}
                    tabIndex={0}
                    aria-expanded={expanded}
                  >
                    <td className={`${STICKY} py-2.5 pr-3 pl-4 transition-colors group-hover:bg-card2 sm:pl-5 ${expanded ? 'bg-card2' : ''}`}>
                      <div className="flex items-center">
                        <span
                          className={`w-7 shrink-0 font-display text-lg leading-none ${top3 ? 'text-accent' : 'text-ink3'}`}
                        >
                          {d.predicted_rank}
                        </span>
                        <TeamDot color={color} />
                        <span className="ml-2.5 font-semibold tracking-wide text-ink">{d.driver_code}</span>
                        <span className="ml-2.5 hidden truncate text-ink3 md:inline">
                          {d.team}
                        </span>
                      </div>
                    </td>
                    {qualiAvailable && (
                      <td className="px-2 py-2.5 text-right text-ink2">
                        <span className="inline-flex items-center justify-end gap-1.5 whitespace-nowrap">
                          {d.grid_penalty > 0 && (
                            <span
                              className="rounded-sm border px-1 py-0.5 text-[10px] font-bold"
                              style={{ color: status.bad, borderColor: `${status.bad}55` }}
                              title={`${d.grid_penalty}-place grid penalty · qualified ${pos(d.quali_pos)}`}
                            >
                              ▼{d.grid_penalty}
                            </span>
                          )}
                          {pos(d.start_pos ?? d.quali_pos)}
                        </span>
                      </td>
                    )}
                    <td className="px-2 py-2.5 sm:px-4">
                      <div className="flex justify-end items-center gap-3 sm:min-w-[7.5rem]">
                        <ProbBar
                          value={(field - d.mean_pos) / Math.max(field - 1, 1)}
                          color={color}
                          className="hidden sm:block"
                        />
                        <span className="w-11 shrink-0 text-right text-ink">
                          P{d.mean_pos.toFixed(1)}
                        </span>
                      </div>
                    </td>
                    <td className={`px-2 py-2.5 text-right ${top3 ? 'text-ink' : 'text-ink2'}`}>
                      {pct(d.win_prob)}
                    </td>
                    <td className="px-2 py-2.5 text-right text-ink2">
                      {pct(d.podium_prob)}
                    </td>
                    <td className="px-2 py-2.5 text-right text-ink2">
                      {pct(d.points_prob)}
                    </td>
                    <td className="px-2 py-2.5 text-center text-ink3 sm:pr-4">
                      <span
                        aria-hidden
                        className={`inline-block transition-transform duration-200 group-hover:text-ink ${expanded ? 'rotate-90 text-accent' : ''}`}
                      >
                        ›
                      </span>
                    </td>
                  </tr>
                  {expanded && (
                    <tr className="border-b border-line/70 last:border-0">
                      <td colSpan={qualiAvailable ? 7 : 6} className="p-0">
                        <div
                          className="sticky left-0"
                          style={viewW != null ? { width: viewW } : undefined}
                        >
                          <WhyRow d={d} />
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
    </Card>
  )
}
