/* Official F1 media headshots, derived from driver names.

   The id convention is first3(first name) + first3(last name) + "01", but the
   media library uses the driver's FULL legal first name, which sometimes
   differs from the broadcast name our API carries — those cases live in
   OVERRIDES (keyed by driver code). The d_driver_fallback_image transformation
   serves a neutral silhouette when a derived id doesn't exist. */

const OVERRIDES: Record<string, string> = {
  // API says "Kimi Antonelli"; F1 media registers "Andrea Kimi Antonelli"
  ANT: 'A/ANDANT01_Andrea%20Kimi_Antonelli/andant01',
}

const BASE =
  'https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/'

export function driverPhotoUrl(name: string, code?: string): string {
  if (code && OVERRIDES[code]) return `${BASE}${OVERRIDES[code]}.png`
  const ascii = name.normalize('NFD').replace(/[̀-ͯ]/g, '')
  const parts = ascii.trim().split(/\s+/)
  const first = parts[0]
  const last = parts[parts.length - 1]
  const id = `${first.slice(0, 3)}${last.slice(0, 3)}`.toUpperCase() + '01'
  return `${BASE}${first[0].toUpperCase()}/${id}_${first}_${last}/${id.toLowerCase()}.png`
}
