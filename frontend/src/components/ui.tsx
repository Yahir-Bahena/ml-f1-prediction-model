import { useRef, type ReactNode } from 'react'
import { label } from '../lib/theme'

/* Shared editorial primitives: flat dark panels, hairlines, Anton for big
   numbers/titles, tiny uppercase tracked labels (`label` from lib/theme). */

/* Little ⓘ that reveals a plain-language explanation on hover, keyboard focus
   or tap. The bubble is display:none until shown (so it never widens the page
   while hidden); on show it's nudged horizontally to stay inside the viewport
   and flipped below the icon when there's no room above (sticky nav). */
export function InfoTip({ text }: { text: string }) {
  const tipRef = useRef<HTMLSpanElement>(null)
  const btnRef = useRef<HTMLButtonElement>(null)

  const place = () => {
    const tip = tipRef.current
    const btn = btnRef.current
    if (!tip || !btn) return
    const b = btn.getBoundingClientRect()
    const vw = document.documentElement.clientWidth
    const w = tip.offsetWidth
    const left = b.left + b.width / 2 - w / 2
    const dx = Math.min(Math.max(left, 8), vw - 8 - w) - left
    tip.style.setProperty('--tip-dx', `${dx}px`)
    const navH = parseFloat(getComputedStyle(document.documentElement).fontSize) * 3.5
    tip.toggleAttribute('data-below', b.top - tip.offsetHeight - 8 < navH)
  }

  return (
    <span className="group/tip relative inline-flex" onPointerEnter={place} onFocus={place}>
      <button
        ref={btnRef}
        type="button"
        aria-label={`What does this mean? ${text}`}
        onClick={(e) => e.currentTarget.focus()} // Safari doesn't focus buttons on tap
        className="inline-flex h-4 w-4 cursor-help items-center justify-center rounded-full text-ink3 transition-colors hover:text-ink focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-cream"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className="h-3.5 w-3.5">
          <circle cx="12" cy="12" r="9" />
          <path d="M12 11v5.5" strokeLinecap="round" />
          <circle cx="12" cy="7.5" r="0.5" fill="currentColor" />
        </svg>
      </button>
      <span
        ref={tipRef}
        role="tooltip"
        className="pointer-events-none absolute hidden bottom-full left-1/2 z-50 mb-2 w-56 max-w-[calc(100vw-16px)] translate-x-[calc(-50%+var(--tip-dx,0px))] rounded-sm border border-line bg-night px-3 py-2 text-left text-xs leading-snug font-normal tracking-normal text-ink2 normal-case shadow-lg shadow-black/50 group-focus-within/tip:block group-hover/tip:block data-below:top-full data-below:bottom-auto data-below:mt-2 data-below:mb-0"
      >
        {text}
      </span>
    </span>
  )
}

export function Card({
  children,
  className = '',
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <div className={`rounded-sm border border-line bg-card ${className}`}>
      {children}
    </div>
  )
}

export function StatTile({
  label: title,
  value,
  sub,
}: {
  label: string
  value: ReactNode
  sub?: string
}) {
  return (
    <Card className="min-w-0 px-4 py-4 sm:px-5">
      <div className={`${label} text-ink3`}>{title}</div>
      <div className="mt-3 font-display text-[clamp(2rem,7vw,2.75rem)] leading-none break-words text-ink">
        {value}
      </div>
      {sub && <div className="mt-2 text-xs text-ink2">{sub}</div>}
    </Card>
  )
}

/** Thin flat magnitude bar on a hairline track; defaults to rust. */
export function ProbBar({
  value,
  color = 'var(--color-rust)',
  className = '',
}: {
  value: number
  color?: string
  className?: string
}) {
  return (
    <div className={`h-1.5 w-full min-w-0 overflow-hidden bg-line ${className}`}>
      <div
        className="h-full"
        style={{ width: `${Math.max(value * 100, value > 0 ? 2 : 0)}%`, background: color }}
      />
    </div>
  )
}

export function TeamDot({ color }: { color: string }) {
  return (
    <span
      className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
      style={{ background: color }}
    />
  )
}

export function Skeleton({ className = '' }: { className?: string }) {
  return (
    <div className={`animate-pulse rounded-sm bg-card2 motion-reduce:animate-none ${className}`} />
  )
}

export function ErrorCard({
  message,
  onRetry,
}: {
  message: string
  onRetry?: () => void
}) {
  return (
    <Card className="px-6 py-10 text-center">
      <div className={`${label} text-accent`}>Something went wrong</div>
      <div className="mt-3 text-sm text-ink2">{message}</div>
      {onRetry && (
        <button
          onClick={onRetry}
          className={`${label} mt-6 inline-flex min-h-11 items-center rounded-full border border-cream/70 px-5 text-cream transition-colors duration-200 hover:bg-cream hover:text-night focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-cream`}
        >
          Retry
        </button>
      )}
    </Card>
  )
}

/* Large Anton headline over a hairline, tiny tracked subtitle; optional rust
   eyebrow pill (same badge as the landing's "Round N"). */
export function PageTitle({
  title,
  sub,
  eyebrow,
}: {
  title: string
  sub?: string
  eyebrow?: string
}) {
  return (
    <div className="mb-8 border-b border-line pb-6 sm:mb-10">
      {eyebrow && (
        <span className={`${label} mb-4 inline-block rounded-full bg-rust px-3 py-1 tracking-[0.18em] text-night`}>
          {eyebrow}
        </span>
      )}
      <h1 className="font-display text-[clamp(2.5rem,10vw,5.5rem)] leading-[0.92] tracking-[-0.01em] break-words text-ink uppercase">
        {title}
      </h1>
      {sub && <p className={`${label} mt-4 max-w-2xl leading-relaxed text-ink2`}>{sub}</p>}
    </div>
  )
}
