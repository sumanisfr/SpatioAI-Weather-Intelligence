import { describe, expect, it } from 'vitest'
import { haversineDistanceKm } from './geo'

describe('haversineDistanceKm', () => {
  it('returns 0 for identical coordinates', () => {
    expect(haversineDistanceKm(21.4, 79.15, 21.4, 79.15)).toBe(0)
  })

  it('calculates realistic distance between Nagpur and Mumbai (~680-720 km)', () => {
    // Nagpur: 21.1458, 79.0882; Mumbai: 19.0760, 72.8777
    const dist = haversineDistanceKm(21.1458, 79.0882, 19.076, 72.8777)
    expect(dist).toBeGreaterThan(680)
    expect(dist).toBeLessThan(730)
  })

  it('handles equator and prime meridian points accurately', () => {
    // 1 degree along equator is ~111.19 km
    const dist = haversineDistanceKm(0, 0, 0, 1)
    expect(dist).toBeGreaterThan(111.0)
    expect(dist).toBeLessThan(111.5)
  })
})
