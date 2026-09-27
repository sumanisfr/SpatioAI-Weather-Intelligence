export type Bbox = [number, number, number, number]

export function bboxToGeoJson(bbox: Bbox): GeoJSON.Feature<GeoJSON.Polygon> {
  const [minLat, maxLat, minLon, maxLon] = bbox
  return {
    type: 'Feature',
    properties: {},
    geometry: {
      type: 'Polygon',
      coordinates: [[[minLon, minLat], [maxLon, minLat], [maxLon, maxLat], [minLon, maxLat], [minLon, minLat]]],
    },
  }
}
