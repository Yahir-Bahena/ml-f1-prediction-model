import { useEffect, useState } from 'react'
import { label } from '../lib/theme'

function parts(target: string) {
  const diff = Math.max(0, new Date(target).getTime() - Date.now())
  const s = Math.floor(diff / 1000)
  return {
    days: Math.floor(s / 86400),
    hours: Math.floor((s % 86400) / 3600),
    mins: Math.floor((s % 3600) / 60),
    secs: s % 60,
  }
}

/* Editorial countdown: four Anton numerals separated by hairlines, tiny
   tracked unit labels underneath. Fills its container width on phones. */
export default function Countdown({ target }: { target: string }) {
  const [t, setT] = useState(() => parts(target))

  useEffect(() => {
    const id = setInterval(() => setT(parts(target)), 1000)
    return () => clearInterval(id)
  }, [target])

  const cells: [number, string][] = [
    [t.days, 'days'],
    [t.hours, 'hrs'],
    [t.mins, 'min'],
    [t.secs, 'sec'],
  ]
  return (
    <div
      className="grid w-full grid-cols-4 divide-x divide-line border-y border-line sm:w-auto"
      role="timer"
      aria-label={`${t.days} days ${t.hours} hours ${t.mins} minutes until lights out`}
    >
      {cells.map(([v, unit]) => (
        <div key={unit} className="px-3 py-3 text-center sm:min-w-20 sm:px-5" aria-hidden>
          <div
            className="font-display text-4xl leading-none text-ink sm:text-5xl"
            style={{ fontVariantNumeric: 'tabular-nums' }}
          >
            {String(v).padStart(2, '0')}
          </div>
          <div className={`${label} mt-2 text-ink3`}>{unit}</div>
        </div>
      ))}
    </div>
  )
}
