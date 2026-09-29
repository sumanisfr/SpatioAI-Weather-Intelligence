import { describe, it, expect } from 'vitest'
import { FALLBACK_EVENTS, FALLBACK_TRACKS, FALLBACK_HEALTH, createFallbackRisk } from './fallbackData'

describe('fallbackData tests', () => {
  it('contains valid mock events and tracks for offline demonstration', () => {
    expect(FALLBACK_EVENTS.length).toBeGreaterThan(0)
    expect(FALLBACK_TRACKS.length).toBeGreaterThan(0)
    expect(FALLBACK_HEALTH.status).toBe('ok')
  })

  it('generates consistent fallback risk structures with GeoJSON polygon', () => {
    const event = FALLBACK_EVENTS[0]
    const risk = createFallbackRisk(event)

    expect(risk.event_id).toBe(event.event_id)
    expect(risk.geojson.type).toBe('FeatureCollection')
    expect(risk.geojson.features.length).toBeGreaterThan(0)
    expect(risk.severity_index).toBeGreaterThan(0)
    expect(risk.max_exceedance_probability).toBeGreaterThan(0)
  })
})
