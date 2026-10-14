import { Route, Routes } from 'react-router-dom'
import Nav from './components/Nav'
import Dashboard from './pages/Dashboard'
import DriverDetail from './pages/DriverDetail'
import Drivers from './pages/Drivers'
import Landing from './pages/Landing'
import ResultDetail from './pages/ResultDetail'
import Results from './pages/Results'
import Standings from './pages/Standings'
import TrackDetail from './pages/TrackDetail'
import Tracks from './pages/Tracks'

/* Standard pages share the centered dark container; the landing hero is
   full-bleed. overflow-x-clip (not hidden — keeps the sticky nav working) and
   min-w-0 stop any page from creating horizontal page scroll on phones; wide
   tables should scroll inside their own overflow-x-auto wrapper. */
function Page({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-[calc(100svh-var(--nav-h))] overflow-x-clip bg-surface">
      <main className="mx-auto w-full max-w-6xl min-w-0 px-4 pt-8 pb-16 sm:px-6 sm:pt-12 sm:pb-24">
        {children}
      </main>
    </div>
  )
}

export default function App() {
  return (
    <div className="min-h-screen bg-surface text-ink">
      <Nav />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/dashboard" element={<Page><Dashboard /></Page>} />
        <Route path="/drivers" element={<Page><Drivers /></Page>} />
        <Route path="/drivers/:code" element={<Page><DriverDetail /></Page>} />
        <Route path="/tracks" element={<Page><Tracks /></Page>} />
        <Route path="/tracks/:location" element={<Page><TrackDetail /></Page>} />
        <Route path="/standings" element={<Page><Standings /></Page>} />
        <Route path="/results" element={<Page><Results /></Page>} />
        <Route path="/results/:season/:round" element={<Page><ResultDetail /></Page>} />
      </Routes>
    </div>
  )
}
