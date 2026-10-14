import type { WeatherForecast } from '../lib/api'

/**
 * Compact race-day rain forecast tag for the next-race hero.
 *
 * Tiers (thresholds on forecast rain probability):
 *  - < 30%: neutral informational tag ("dry race expected")
 *  - 30-49%: notable - sky-tinted tag ("showers possible")
 *  - >= 50%: prominent - filled sky tag, "wet race conditions likely", with a
 *    note that the model's predictions already account for it (is_wet input).
 * Renders nothing when the forecast is missing (API null) or has no
 * probability - no empty shell.
 *
 * Accessibility: meaning is carried by icon + text, never color alone;
 * icons are inline SVG (no emoji), aria-hidden since the text says it all.
 */
export default function WeatherChip({ forecast }: { forecast: WeatherForecast | null }) {
  if (!forecast || forecast.rain_prob === null) return null

  const p = forecast.rain_prob
  const pctLabel = `${Math.round(p * 100)}% rain`

  let cls: string
  let text: string
  let icon: 'sun' | 'drizzle' | 'rain'
  if (p >= 0.5) {
    cls = 'border-sky-300/60 bg-sky-400/15 text-sky-100'
    text = `${pctLabel} — wet race conditions likely · factored into predictions`
    icon = 'rain'
  } else if (p >= 0.3) {
    cls = 'border-sky-400/40 text-sky-200'
    text = `${pctLabel} — showers possible`
    icon = 'drizzle'
  } else {
    cls = 'border-line text-ink2'
    text = `${pctLabel} — dry race expected`
    icon = 'sun'
  }

  return (
    <span
      className={`inline-flex max-w-full items-center gap-2 rounded-sm border px-2.5 py-1.5 text-[11px] leading-snug font-medium tracking-[0.12em] uppercase ${cls}`}
    >
      <WeatherIcon kind={icon} />
      <span className="min-w-0">{text}</span>
    </span>
  )
}

function WeatherIcon({ kind }: { kind: 'sun' | 'drizzle' | 'rain' }) {
  const common = {
    width: 14,
    height: 14,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.8,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    'aria-hidden': true,
    className: 'shrink-0',
  }
  if (kind === 'sun') {
    return (
      <svg {...common}>
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
      </svg>
    )
  }
  if (kind === 'drizzle') {
    return (
      <svg {...common}>
        <path d="M20 16.6A5 5 0 0 0 18 7h-1.3A7 7 0 1 0 6 15.7" />
        <path d="M8 19v2m4-3v2m4-3v2" />
      </svg>
    )
  }
  return (
    <svg {...common}>
      <path d="M20 16.6A5 5 0 0 0 18 7h-1.3A7 7 0 1 0 6 15.7" />
      <path d="M8 17v4m4-3v4m4-5v4" />
    </svg>
  )
}
