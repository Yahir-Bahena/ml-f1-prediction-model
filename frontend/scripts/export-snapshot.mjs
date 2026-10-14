// Freezes the live API into static JSON under public/data so the site can be
// hosted with no backend (e.g. Vercel). Run with the API up on :8000:
//   npm run snapshot
// The production build reads these files instead of calling /api (see
// src/lib/api.ts and .env.production).
import { mkdir, rm, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const API = process.env.API_URL ?? 'http://localhost:8000/api'
const OUT = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'data')

async function getJson(path) {
  const res = await fetch(API + path)
  if (!res.ok) throw new Error(`${path} -> ${res.status}`)
  return res.json()
}

async function save(path, data) {
  const file = join(OUT, `${path}.json`)
  await mkdir(dirname(file), { recursive: true })
  await writeFile(file, JSON.stringify(data))
}

await rm(OUT, { recursive: true, force: true })

const drivers = await getJson('/drivers')
const tracks = await getJson('/tracks')
const results = await getJson('/results')
const nextRace = await getJson('/next-race')

await save('next-race', nextRace)
await save('predictions/next', await getJson('/predictions/next'))
await save('model/metrics', await getJson('/model/metrics'))
await save('schedule', await getJson('/schedule'))
await save('standings', await getJson('/standings'))
await save('drivers', drivers)
await save('tracks', tracks)
await save('results', results)

for (const d of drivers) {
  await save(`drivers/${d.driver_code}/stats`, await getJson(`/drivers/${d.driver_code}/stats`))
}
for (const t of tracks) {
  await save(`tracks/${t.location}/stats`, await getJson(`/tracks/${encodeURIComponent(t.location)}/stats`))
}
for (const r of results) {
  await save(`results/${r.season}/${r.round}`, await getJson(`/results/${r.season}/${r.round}`))
}

await save('meta', { generated_at: new Date().toISOString(), next_race: nextRace.name })
console.log(`Snapshot written to ${OUT}`)
