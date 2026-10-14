import { useState, type ReactNode } from 'react'
import type { DriverPrediction } from '../lib/api'
import { driverPhotoUrl } from '../lib/driverPhoto'
import { pct, teamHex } from '../lib/format'
import { label, status } from '../lib/theme'
import { InfoTip } from './ui'

const PROJ_INFO =
  "Average finishing position across 10,000 simulated runs — lower is better. The podium is ranked by this, which rewards consistency and reliability, not raw pace. When two drivers' projected finishes are within ~0.25 of a position (effectively tied), the higher win % is ranked ahead — so a clear win favourite won't sit below a near-identical but less likely winner. See 'Win' for who is fastest."
const WIN_INFO =
  'Chance of finishing 1st across the simulations. Driven by raw pace, so the fastest car can lead here even when a more reliable car has the better projected finish.'

/** "Max Verstappen" → ["Max", "Verstappen"]; single names go in the big line. */
function splitName(name: string): [string, string] {
  const i = name.indexOf(' ')
  return i === -1 ? ['', name] : [name.slice(0, i), name.slice(i + 1)]
}

/* Driver cut-out standing on the card's bottom edge; falls back to the code. */
function Photo({ d, className }: { d: DriverPrediction; className: string }) {
  const [ok, setOk] = useState(true)
  return ok ? (
    <img
      src={driverPhotoUrl(d.driver_name, d.driver_code)}
      alt={d.driver_name}
      loading="lazy"
      onError={() => setOk(false)}
      className={`object-contain object-bottom ${className}`}
    />
  ) : (
    <div className={`flex items-end justify-center ${className}`}>
      <span className="mb-3 font-display text-4xl text-ink3 uppercase">{d.driver_code}</span>
    </div>
  )
}

function Stat({
  title,
  info,
  children,
  big = false,
}: {
  title: string
  info?: ReactNode
  children: ReactNode
  big?: boolean
}) {
  return (
    <div className="min-w-0">
      <div className={`${label} flex items-center gap-1 text-ink3`}>
        {title}
        {info}
      </div>
      <div
        className={`mt-2 flex items-baseline gap-1.5 font-display leading-none ${
          big ? 'text-5xl text-accent sm:text-6xl' : 'text-3xl text-ink'
        }`}
        style={{ fontVariantNumeric: 'tabular-nums' }}
      >
        {children}
      </div>
    </div>
  )
}

function StartValue({ d }: { d: DriverPrediction }) {
  return (
    <>
      {d.start_pos != null ? `P${Math.round(d.start_pos)}` : '—'}
      {d.grid_penalty > 0 && (
        <span className="font-sans text-xs font-bold" style={{ color: status.bad }}>
          ▼{d.grid_penalty}
        </span>
      )}
    </>
  )
}

function startInfo(d: DriverPrediction) {
  return d.grid_penalty > 0 ? (
    <InfoTip
      text={`Starts P${Math.round(d.start_pos ?? 0)} after a ${d.grid_penalty}-place grid penalty (qualified P${d.quali_pos}).`}
    />
  ) : undefined
}

/* Team colour as a flat 3px rule along the top edge — the only colour flourish. */
function TeamRule({ color }: { color: string }) {
  return <div aria-hidden className="absolute inset-x-0 top-0 h-[3px]" style={{ background: color }} />
}

function Winner({ d }: { d: DriverPrediction }) {
  const color = teamHex(d.team_color)
  const [first, last] = splitName(d.driver_name)
  return (
    <div className="rise relative flex min-w-0 flex-col rounded-sm border border-line bg-card" style={{ animationDelay: '60ms' }}>
      <TeamRule color={color} />
      {/* name block + photo; the photo layer clips itself so tooltips below can overflow the card */}
      <div className="relative grid grid-cols-[minmax(0,1fr)_auto] items-end gap-2 px-5 pt-6 sm:px-7 sm:pt-7">
        <div aria-hidden className="pointer-events-none absolute top-3 right-4 font-display text-[7rem] leading-none text-ink/[0.05] select-none sm:text-[10rem]">
          {d.driver_number}
        </div>
        <div className="relative min-w-0 pb-5 sm:pb-7">
          <div className="flex items-center gap-3">
            <span className={`${label} rounded-full bg-rust px-3 py-1 tracking-[0.18em] text-night`}>P1</span>
            <span className={`${label} text-ink3`}>Predicted winner</span>
          </div>
          {first && <div className={`${label} mt-5 text-ink2 sm:mt-7`}>{first}</div>}
          <div className="mt-1 font-display text-[clamp(2.75rem,9vw,5.75rem)] leading-[0.88] break-words text-ink uppercase">
            {last}
          </div>
          <div className={`${label} mt-3 flex items-center gap-2 text-ink2`}>
            <span className="inline-block h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />
            <span className="truncate">{d.team}</span>
          </div>
        </div>
        <Photo d={d} className="relative h-36 w-28 sm:h-56 sm:w-44 lg:h-64 lg:w-52" />
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-5 border-t border-line px-5 py-5 sm:grid-cols-4 sm:px-7">
        <Stat title="Win" info={<InfoTip text={WIN_INFO} />} big>
          {pct(d.win_prob)}
        </Stat>
        <Stat title="Proj. finish" info={<InfoTip text={PROJ_INFO} />}>
          P{d.mean_pos.toFixed(1)}
        </Stat>
        <Stat title="Podium">{pct(d.podium_prob)}</Stat>
        <Stat title="Start" info={startInfo(d)}>
          <StartValue d={d} />
        </Stat>
      </div>
    </div>
  )
}

function Runner({ d, place }: { d: DriverPrediction; place: number }) {
  const color = teamHex(d.team_color)
  const [first, last] = splitName(d.driver_name)
  return (
    <div
      className="rise relative flex min-w-0 flex-col rounded-sm border border-line bg-card"
      style={{ animationDelay: `${place * 90}ms` }}
    >
      <TeamRule color={color} />
      <div className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-end gap-4 px-5 pt-5">
        <div className="self-start font-display text-5xl leading-none text-ink3">{place}</div>
        <div className="min-w-0 pb-4">
          {first && <div className={`${label} text-ink2`}>{first}</div>}
          <div className="mt-1 font-display text-[clamp(1.75rem,6vw,2.25rem)] leading-[0.9] md:text-[1.75rem] lg:text-[2.25rem] break-words text-ink uppercase">
            {last}
          </div>
          <div className={`${label} mt-2 flex items-center gap-2 text-ink3`}>
            <span className="inline-block h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: color }} />
            <span className="truncate">{d.team}</span>
          </div>
        </div>
        <Photo d={d} className="h-24 w-20 sm:h-28 sm:w-24" />
      </div>
      <div className="mt-auto grid grid-cols-4 gap-3 border-t border-line px-5 py-4">
        <MiniStat title="Win" info={<InfoTip text={WIN_INFO} />}>
          <span className="text-accent">{pct(d.win_prob)}</span>
        </MiniStat>
        <MiniStat title="Proj." info={<InfoTip text={PROJ_INFO} />}>
          P{d.mean_pos.toFixed(1)}
        </MiniStat>
        <MiniStat title="Podium">{pct(d.podium_prob)}</MiniStat>
        <MiniStat title="Start" info={startInfo(d)}>
          <StartValue d={d} />
        </MiniStat>
      </div>
    </div>
  )
}

function MiniStat({ title, info, children }: { title: string; info?: ReactNode; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="flex items-center gap-0.5 text-[10px] font-medium tracking-[0.16em] text-ink3 uppercase">
        {title}
        {info}
      </div>
      <div
        className="mt-1.5 flex items-baseline gap-1 font-display text-xl leading-none text-ink"
        style={{ fontVariantNumeric: 'tabular-nums' }}
      >
        {children}
      </div>
    </div>
  )
}

/* Editorial podium: the predicted winner as a large feature panel, P2/P3
   stacked beside it (below it on phones and tablets). */
export default function PodiumHero({ top3 }: { top3: DriverPrediction[] }) {
  if (top3.length < 3) return null
  const [p1, p2, p3] = top3
  return (
    <div className="grid gap-3 lg:grid-cols-[minmax(0,1.65fr)_minmax(0,1fr)]">
      <Winner d={p1} />
      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-1">
        <Runner d={p2} place={2} />
        <Runner d={p3} place={3} />
      </div>
    </div>
  )
}
