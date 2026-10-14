/* Shared design constants for places that can't use Tailwind token utilities
   — mostly Recharts, whose SVG props need literal colours (presentation
   attributes don't resolve CSS vars reliably), and inline style={{}} colours.
   Values mirror the @theme block in src/index.css; keep the two in sync.

   Usage:
     import { chart, status, label } from '../lib/theme'
     <CartesianGrid stroke={chart.grid} vertical={false} />
     <XAxis tick={chart.tick} axisLine={{ stroke: chart.grid }} tickLine={false} />
     <Tooltip {...chart.tooltip} />
     <Line stroke={chart.series[0]} dot={{ fill: ..., stroke: chart.dotRing }} />
     <span style={{ color: status.good }}>…</span>
     <p className={`${label} text-ink3`}>Tiny tracked caption</p>
*/

export const tokens = {
  surface: '#0d0d0c',
  card: '#151514',
  card2: '#1e1e1c',
  line: '#2b2b29',
  ink: '#fdf0e6',
  ink2: '#bab0a9',
  ink3: '#918a84',
  accent: '#ff6a13', // bright orange, legible as text on dark (~7:1 on surface)
  accent2: '#ff9248',
  rust: '#ff6a13', // solid fills / pills (use night text on top)
  cream: '#fdf0e6',
  night: '#0d0d0c',
} as const

/** Tiny uppercase tracked label — the metadata style used across the site. */
export const label = 'text-[11px] font-medium tracking-[0.2em] uppercase'

/** Semantic status colours tuned for dark panels (all ≥6:1 on card2). */
export const status = {
  good: '#4ade80',
  warn: '#fbbf24',
  bad: '#f87171',
} as const

export const chart = {
  grid: tokens.line,
  axis: tokens.line,
  /** spread into XAxis/YAxis `tick` */
  tick: { fill: tokens.ink3, fontSize: 11 },
  /** stroke around dots so they cut out of lines on the card background */
  dotRing: tokens.card,
  /** cursor/hover rule */
  cursor: { stroke: tokens.ink3, strokeWidth: 1, strokeDasharray: '3 3' },
  /** spread into <Tooltip {...chart.tooltip} /> */
  tooltip: {
    contentStyle: {
      background: tokens.night,
      border: `1px solid ${tokens.line}`,
      borderRadius: 2,
      fontSize: 12,
      color: tokens.ink,
    },
    labelStyle: {
      color: tokens.ink3,
      fontSize: 11,
      letterSpacing: '0.15em',
      textTransform: 'uppercase' as const,
    },
    itemStyle: { color: tokens.ink },
  },
  /** non-team series order: rust first, then cream and muted neutrals */
  series: [tokens.accent, tokens.cream, tokens.ink3, tokens.accent2, tokens.ink2],
} as const
