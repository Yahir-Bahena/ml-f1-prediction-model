import { useEffect, useRef, useState } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'

const links = [
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/drivers', label: 'Drivers' },
  { to: '/tracks', label: 'Tracks' },
  { to: '/standings', label: 'Standings' },
  { to: '/results', label: 'Track Record' },
]

const focusRing =
  'focus-visible:rounded-sm focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-cream'
// tiny tracked uppercase shared by every nav label
const label = 'text-[11px] font-medium tracking-[0.2em] whitespace-nowrap uppercase'

/* Desktop link: muted cream → full cream on hover/active, with a short rust
   rule under the active route. */
function DeskLink({
  to,
  muted = 'text-cream/75',
  children,
}: {
  to: string
  muted?: string
  children: React.ReactNode
}) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `${label} ${focusRing} relative py-1 transition-colors duration-200 hover:text-cream ${
          isActive
            ? 'text-cream after:absolute after:inset-x-0 after:-bottom-1 after:h-0.5 after:bg-rust'
            : muted
        }`
      }
    >
      {children}
    </NavLink>
  )
}

/* One nav for every route. On the landing it's transparent and overlaid on
   the hero photo (absolute, roomier padding); elsewhere it's a solid sticky
   night bar of height --nav-h with a hairline bottom border. Below lg the
   left links collapse into a menu button. */
export default function Nav() {
  const { pathname } = useLocation()
  const landing = pathname === '/'
  // Mobile menu: full-screen night panel under the bar with big Anton links.
  // Open state is keyed to the pathname it was opened on, so any navigation
  // closes it without a setState-in-effect.
  const [openAt, setOpenAt] = useState<string | null>(null)
  const open = openAt === pathname
  const close = () => setOpenAt(null)
  // landing keeps its original brighter links over the photo
  const muted = landing ? 'text-cream/85' : 'text-cream/70'
  const toggleRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)

  // while open: lock body scroll, close on Escape, keep Tab inside bar+panel,
  // and move focus to the first menu link; focus returns to the toggle on close
  useEffect(() => {
    if (!open) return
    const toggle = toggleRef.current
    const prevOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    panelRef.current?.querySelector<HTMLElement>('a')?.focus()

    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpenAt(null)
        return
      }
      if (e.key !== 'Tab' || !toggle || !panelRef.current) return
      const items = [toggle, ...panelRef.current.querySelectorAll<HTMLElement>('a')]
      const first = items[0]
      const last = items[items.length - 1]
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault()
        last.focus()
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', onKey)
    return () => {
      document.body.style.overflow = prevOverflow
      document.removeEventListener('keydown', onKey)
      toggle?.focus({ preventScroll: true })
    }
  }, [open])

  // open menu pins the bar (fixed) and gives it a solid background even on
  // the landing, so the toggle stays reachable above the panel
  const position = open
    ? 'fixed bg-night'
    : landing
      ? 'absolute'
      : 'sticky border-b border-line bg-night/95 backdrop-blur-sm'

  return (
    <>
      <header className={`${position} inset-x-0 top-0 z-40`}>
        <div
          className={`grid grid-cols-[1fr_auto_1fr] items-center gap-4 lg:gap-6 ${
            landing ? 'px-5 py-5 sm:px-8' : 'h-[var(--nav-h)] px-4 sm:px-6 lg:px-8'
          }`}
        >
          <div className="flex items-center">
            <button
              ref={toggleRef}
              type="button"
              onClick={() => setOpenAt(open ? null : pathname)}
              aria-expanded={open}
              aria-controls="site-menu"
              aria-label={open ? 'Close menu' : 'Open menu'}
              className={`-ml-2.5 flex h-11 w-11 items-center justify-center text-cream lg:hidden ${focusRing}`}
            >
              {/* two hairlines that cross into an X */}
              <span aria-hidden className="relative block h-3 w-5">
                <span
                  className={`absolute left-0 h-px w-5 bg-current transition-transform duration-200 motion-reduce:transition-none ${
                    open ? 'top-1.5 rotate-45' : 'top-0.5'
                  }`}
                />
                <span
                  className={`absolute left-0 h-px w-5 bg-current transition-transform duration-200 motion-reduce:transition-none ${
                    open ? 'top-1.5 -rotate-45' : 'top-2.5'
                  }`}
                />
              </span>
            </button>
            <nav className="hidden items-center gap-7 lg:flex" aria-label="Primary">
              {links.map((l) => (
                <DeskLink key={l.to} to={l.to} muted={muted}>
                  {l.label}
                </DeskLink>
              ))}
            </nav>
          </div>

          <Link
            to="/"
            className={`text-lg font-semibold tracking-tight whitespace-nowrap text-cream lowercase ${focusRing}`}
          >
            f1 predictor
          </Link>

          <div className="flex justify-self-end">
            <DeskLink to="/dashboard" muted={muted}>Predictions</DeskLink>
          </div>
        </div>
      </header>

      {open && (
        <div
          id="site-menu"
          ref={panelRef}
          role="dialog"
          aria-modal="true"
          aria-label="Site menu"
          className="fixed inset-0 z-30 flex flex-col overflow-y-auto bg-night px-4 pt-[calc(var(--nav-h)+1.5rem)] pb-8 sm:px-6 lg:hidden"
        >
          <nav aria-label="Primary" className="border-t border-cream/15">
            <ul>
              {links.map((l, i) => (
                <li key={l.to} className="fade-up border-b border-cream/15" style={{ animationDelay: `${i * 40}ms` }}>
                  <NavLink
                    to={l.to}
                    onClick={close}
                    className={({ isActive }) =>
                      `flex min-h-11 items-baseline gap-4 py-3 transition-colors duration-200 hover:text-cream ${focusRing} ${
                        isActive ? 'text-cream' : 'text-cream/60'
                      }`
                    }
                  >
                    {({ isActive }) => (
                      <>
                        <span className={`${label} w-6 shrink-0 text-cream/60`}>
                          {String(i + 1).padStart(2, '0')}
                        </span>
                        <span className="font-display text-[clamp(2.25rem,11vw,3.5rem)] leading-none uppercase">
                          {l.label}
                        </span>
                        {isActive && (
                          <span aria-hidden className="ml-auto h-2 w-2 shrink-0 self-center rounded-full bg-rust" />
                        )}
                      </>
                    )}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
          <p className="mt-auto pt-10 text-[11px] font-medium tracking-[0.2em] text-cream/60 uppercase">
            10k simulations per race
          </p>
        </div>
      )}
    </>
  )
}
