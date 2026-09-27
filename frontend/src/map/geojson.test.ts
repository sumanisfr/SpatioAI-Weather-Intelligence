import { describe, expect, it } from 'vitest'
import { bboxToGeoJson } from './geojson'

describe('bboxToGeoJson', () => {
  it('keeps GeoJSON coordinates in longitude, latitude order', () => {
    const feature = bboxToGeoJson([10, 20, 70, 80])
    expect(feature.geometry.coordinates[0][0]).toEqual([70, 10])
    expect(feature.geometry.coordinates[0][2]).toEqual([80, 20])
  })
})
