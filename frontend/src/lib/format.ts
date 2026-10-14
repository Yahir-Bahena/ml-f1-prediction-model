/** Normalize fastf1's team color (hex without '#', possibly empty). */
export function teamHex(color: string | null | undefined): string {
  if (color && /^[0-9a-fA-F]{6}$/.test(color)) return `#${color}`
  return '#8a8fa3'
}

export function pct(x: number): string {
  const v = x * 100
  if (v >= 99.5) return '>99%'
  if (v < 1 && v > 0) return '<1%'
  return `${v >= 9.5 ? Math.round(v) : v.toFixed(1)}%`
}

export function pos(n: number | null | undefined): string {
  return n == null ? '—' : `P${Math.round(n)}`
}

export function num(n: number | null | undefined, digits = 1): string {
  return n == null ? '—' : n.toFixed(digits)
}

// Race times are shown in US Eastern time. `timeZoneName: 'short'` appends the
// correct label automatically — EDT during daylight saving (summer), EST in
// winter — so it is always accurate rather than a hard-coded abbreviation.
const ET = 'America/New_York'

export function etDateTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    timeZone: ET,
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZoneName: 'short',
  })
}

export function etDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    timeZone: ET,
    weekday: 'short',
    month: 'short',
    day: 'numeric',
  })
}

export function etTime(iso: string): string {
  return new Date(iso).toLocaleTimeString(undefined, {
    timeZone: ET,
    hour: '2-digit',
    minute: '2-digit',
    timeZoneName: 'short',
  })
}
