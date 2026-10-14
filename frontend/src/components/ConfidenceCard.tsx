import type { Confidence } from '../lib/api'
import { label as labelCls, status } from '../lib/theme'
import { Card } from './ui'

const LEVEL = {
  high: { label: 'High confidence', color: status.good },
  medium: { label: 'Medium confidence', color: status.warn },
  low: { label: 'Low confidence', color: status.bad },
} as const

function ReasonIcon({ kind }: { kind: 'good' | 'warn' }) {
  return kind === 'good' ? (
    <svg viewBox="0 0 24 24" fill="none" stroke={status.good} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5 shrink-0" aria-hidden>
      <path d="m5 13 4 4L19 7" />
    </svg>
  ) : (
    <svg viewBox="0 0 24 24" fill="none" stroke={status.warn} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5 shrink-0" aria-hidden>
      <path d="M12 4 2.5 20h19L12 4ZM12 10v4.5M12 17.5v.1" />
    </svg>
  )
}

/* Race-level model confidence: the calibrated win probability of the top pick,
   with the factors pushing it up or down. */
export default function ConfidenceCard({ confidence }: { confidence: Confidence }) {
  const { label, color } = LEVEL[confidence.level]
  const good = confidence.reasons.filter((r) => r.kind === 'good')
  const warn = confidence.reasons.filter((r) => r.kind === 'warn')

  return (
    <Card className="rise">
      <div className="grid md:grid-cols-[280px_1fr]">
        <div className="border-b border-line px-5 py-5 sm:px-6 md:border-r md:border-b-0">
          <div className={`${labelCls} text-ink3`}>Model confidence</div>
          <div className="mt-4 flex items-end justify-between gap-4">
            <div className="font-display text-2xl leading-none uppercase" style={{ color }}>
              {label}
            </div>
            <span
              className="font-display text-5xl leading-[0.85] text-ink"
              style={{ fontVariantNumeric: 'tabular-nums' }}
            >
              {confidence.score}%
            </span>
          </div>
          <div className="mt-4 h-1.5 overflow-hidden bg-line">
            <div
              className="bar-fill h-full"
              style={{ width: `${confidence.score}%`, background: color }}
            />
          </div>
          <p className="mt-3 text-xs leading-relaxed text-ink3">
            Calibrated probability the predicted winner actually wins — at this
            level, past predictions came true about {confidence.score}% of the time.
          </p>
        </div>

        <div className="grid gap-x-8 gap-y-5 px-5 py-5 sm:grid-cols-2 sm:px-6 md:grid-cols-1 lg:grid-cols-2">
          {good.length > 0 && (
            <div>
              <div className={`${labelCls} mb-3 text-ink3`}>Working for the call</div>
              <ul className="space-y-2">
                {good.map((r) => (
                  <li key={r.text} className="flex items-start gap-2.5 text-sm leading-snug text-ink2">
                    <span className="mt-0.5"><ReasonIcon kind="good" /></span>
                    {r.text}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {warn.length > 0 && (
            <div>
              <div className={`${labelCls} mb-3 text-ink3`}>Against it</div>
              <ul className="space-y-2">
                {warn.map((r) => (
                  <li key={r.text} className="flex items-start gap-2.5 text-sm leading-snug text-ink2">
                    <span className="mt-0.5"><ReasonIcon kind="warn" /></span>
                    {r.text}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </Card>
  )
}
