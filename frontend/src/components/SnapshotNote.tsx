import { useEffect, useState } from 'react'
import { label } from '../lib/theme'

/* Static builds only: the site is a frozen copy of the model's output, so say
   when it was taken. Renders nothing in dev (live API) or if meta is missing. */
export default function SnapshotNote() {
  const [when, setWhen] = useState<string | null>(null)

  useEffect(() => {
    if (import.meta.env.VITE_STATIC !== 'true') return
    fetch('/data/meta.json')
      .then((r) => r.json())
      .then((m: { generated_at: string }) =>
        setWhen(
          new Date(m.generated_at).toLocaleDateString('en-US', {
            month: 'long',
            day: 'numeric',
            year: 'numeric',
          }),
        ),
      )
      .catch(() => {})
  }, [])

  if (!when) return null
  return (
    <footer className="border-t border-line bg-night px-4 py-6 text-center">
      <p className={`${label} text-ink3`}>Data snapshot as of {when}</p>
    </footer>
  )
}
