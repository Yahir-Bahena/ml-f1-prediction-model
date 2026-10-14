import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../lib/api'
import { etDate, etTime } from '../lib/format'
import { useFetch } from '../lib/useFetch'

/* Eased count-up for the headline stat; snaps instantly under reduced motion. */
function useCountUp(target: number | null, ms = 1100) {
  const [value, setValue] = useState(0)
  useEffect(() => {
    if (target == null) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setValue(target)
      return
    }
    let raf: number
    const t0 = performance.now()
    const step = (t: number) => {
      const p = Math.min(1, (t - t0) / ms)
      setValue(target * (1 - Math.pow(1 - p, 3)))
      if (p < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [target, ms])
  return value
}

/* Monochrome inline marks so the marquee needs no external assets.
   Model/prediction stack only — the frontend chrome doesn't belong here. */
const STACK: { name: string; icon: React.ReactNode }[] = [
  {
    name: 'Python',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
        <path d="M12 3c-2.5 0-4 .9-4 2.6V8h8v1.3H6.2C4.4 9.3 3 10.8 3 13s1.4 3.7 3.2 3.7H8v-2.4c0-1.9 1.6-3.3 3.6-3.3h4.8c1.6 0 2.6-1.1 2.6-2.7V5.6C19 3.9 14.5 3 12 3Z" />
        <path d="M12 21c2.5 0 4-.9 4-2.6V16H8v-1.3h9.8c1.8 0 3.2-1.5 3.2-3.7" />
        <circle cx="9.6" cy="5.9" r="0.9" fill="currentColor" stroke="none" />
        <circle cx="14.4" cy="18.1" r="0.9" fill="currentColor" stroke="none" />
      </svg>
    ),
  },
  {
    name: 'FastAPI',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round">
        <circle cx="12" cy="12" r="9" />
        <path d="M13.5 5.5 8 13h3.5L10.5 18.5 16 11h-3.5l1-5.5Z" />
      </svg>
    ),
  },
  {
    name: 'LightGBM',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
        <path d="M12 21V9M12 9 7 4M12 9l5-5M12 15l-4-4M12 15l4-4" />
      </svg>
    ),
  },
  {
    name: 'FastF1',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M5 21V4h13l-2.5 4H18l-2.5 4H8" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    name: 'pandas',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
        <path d="M8 3v13M8 19.5v1.5M12 8v13M12 3v1.5M16 3v13M16 19.5v1.5" />
      </svg>
    ),
  },
  {
    name: 'NumPy',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round">
        <path d="m12 2 8 4.5v9L12 20l-8-4.5v-9L12 2Z" />
        <path d="M12 11 4 6.5M12 11l8-4.5M12 11v9" />
      </svg>
    ),
  },
  {
    name: 'SciPy',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
        <circle cx="12" cy="12" r="8.5" />
        <path d="M4.5 14.5c2.5-1.5 4-4.5 7.5-4.5s5 3 7.5 1.5" />
        <circle cx="15.5" cy="7.5" r="1.2" fill="currentColor" stroke="none" />
      </svg>
    ),
  },
]

function StackMarquee() {
  const items = [...STACK, ...STACK] // duplicated for the seamless -50% loop
  return (
    <div className="marquee relative overflow-hidden [mask-image:linear-gradient(to_right,transparent,black_8%,black_92%,transparent)]" aria-label="Technology stack">
      <div className="marquee-track flex w-max items-center gap-10 py-1">
        {items.map((item, i) => (
          <span
            key={`${item.name}-${i}`}
            className="flex shrink-0 items-center gap-2.5 text-cream/55 transition-colors duration-200 hover:text-cream"
            aria-hidden={i >= STACK.length}
          >
            <span className="h-5 w-5">{item.icon}</span>
            <span className="text-xs font-medium tracking-[0.2em] uppercase">{item.name}</span>
          </span>
        ))}
      </div>
    </div>
  )
}

/* Shown only if the next-race request fails; while loading the word box is
   reserved but empty so the real location doesn't replace a flash of text. */
const FALLBACK_WORD = 'LIGHTS OUT'

export default function Landing() {
  const metrics = useFetch(api.metrics)
  const race = useFetch(api.nextRace)

  const winnerPct = metrics.data ? metrics.data.model.winner_hit_rate * 100 : null
  const animatedPct = useCountUp(winnerPct)
  const nFeatures = metrics.data?.feature_importance.length

  const word = race.data ? race.data.location.toUpperCase() : race.error ? FALLBACK_WORD : null
  // tiny tracked uppercase used for every label on the page
  const label = 'text-[11px] font-medium tracking-[0.2em] uppercase'

  return (
    <div className="bg-night text-cream">
      {/* hero: full-bleed photo, one giant word, the nav overlays it */}
      <section className="relative flex h-svh min-h-[34rem] flex-col items-center justify-center overflow-hidden">
        <div aria-hidden className="absolute inset-0">
          <img
            src="/f1_background.png"
            alt=""
            className="absolute inset-0 h-full w-full object-cover object-[30%_60%]"
            onError={(e) => ((e.target as HTMLImageElement).style.display = 'none')}
          />
          {/* light wash + vignette: keeps cream type legible on the low-res photo */}
          <div className="absolute inset-0 bg-black/30" />
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_35%,rgba(0,0,0,0.55)_100%)]" />
          {/* melt into the dark section below instead of a hard photo edge */}
          <div className="absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-night to-transparent" />
        </div>

        <div className="relative flex w-full flex-col items-center px-4">

          <div className="hero-word-box flex w-full items-center justify-center">
            {word && (
              <h1
                key={word}
                className="hero-word scale-in font-display whitespace-nowrap tracking-[-0.01em] text-accent"
                style={{ '--chars': Math.max(word.length, 4), animationDelay: '120ms' } as React.CSSProperties}
              >
                {word}
                {race.data && <span className="sr-only"> — next race: {race.data.name}</span>}
              </h1>
            )}
          </div>
        </div>

        {/* bottom info line + CTA, then the scroll hairline */}
        <div
          className="fade-up absolute inset-x-0 bottom-24 flex flex-col items-center gap-5 px-5 text-center sm:bottom-20"
          style={{ animationDelay: '320ms' }}
        >
          <p className={`${label} flex flex-col items-center gap-x-3 gap-y-1.5 text-cream/80 sm:flex-row sm:flex-wrap sm:justify-center`}>
            {race.data ? (
              <>
                <span className="text-cream">{race.data.name}</span>
                <span aria-hidden className="hidden sm:inline">·</span>
                <span>
                  {etDate(race.data.race_start_utc)}, {etTime(race.data.race_start_utc)}
                </span>
              </>
            ) : (
              <span>&nbsp;</span>
            )}
            {winnerPct != null && (
              <>
                <span aria-hidden className="hidden sm:inline">·</span>
                <span>{winnerPct.toFixed(1)}% winners called</span>
              </>
            )}
          </p>
          <Link
            to="/dashboard"
            className={`group inline-flex items-center gap-2 rounded-full border border-cream/70 px-5 py-2.5 text-cream transition-colors duration-200 hover:bg-cream hover:text-night focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-cream ${label}`}
          >
            View predictions
            <span aria-hidden className="transition-transform duration-200 group-hover:translate-x-0.5">
              →
            </span>
          </Link>
        </div>

        <div aria-hidden className="scroll-cue absolute bottom-0 left-1/2 h-12 w-px -translate-x-1/2 overflow-hidden bg-cream/30" />
      </section>

      {/* below the fold: the model in two lines — accuracy stat + the stack */}
      <section className="mx-auto max-w-6xl px-5 py-24 sm:px-8 sm:py-32">
        <div className="grid gap-12 md:grid-cols-[auto_1fr] md:items-end md:gap-20">
          <div>
            <p className={`${label} text-cream/60`}>Winner hit-rate · backtested</p>
            <p className="mt-3 font-display text-[clamp(5rem,18vw,11rem)] leading-[0.9] text-cream">
              {winnerPct != null ? `${animatedPct.toFixed(1)}%` : '—'}
            </p>
          </div>
          <div className="md:pb-3">
            <p className="max-w-md text-base leading-relaxed text-cream/75">
              A LightGBM model trained on every lap raced since 2022 — qualifying pace, race trim,
              weather and form — simulated 10,000 times per race into live win probabilities.
            </p>
            <p className={`${label} mt-5 text-cream/60`}>
              {metrics.data?.model.races ?? '—'} races · {nFeatures ?? '—'} features · 10k sims
            </p>
          </div>
        </div>

        <div className="mt-20 border-t border-cream/15 pt-8">
          <p className={`${label} mb-6 text-cream/60`}>Built with</p>
          <StackMarquee />
        </div>
      </section>
    </div>
  )
}
