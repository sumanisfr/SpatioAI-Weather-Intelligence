import type { Event, Track, RiskResponse, HealthResponse } from '../types/api'

export const FALLBACK_HEALTH: HealthResponse = {
  status: 'ok',
  service: 'Sanket',
  version: '1.0.0',
  models_loaded: true,
  models: {
    gnn: { available: true, loaded: true, path: 'models/spatiotemporal_gnn_best.pt', error: null },
    unet: { available: true, loaded: true, path: 'models/downscaler_unet_best.pt', error: null },
    diffusion: { available: true, loaded: true, path: 'models/diffusion_downscaler_best.pt', error: null },
    physics_diffusion: { available: true, loaded: true, path: 'models/diffusion_physics_ablation.pt', error: null },
  },
}

export const FALLBACK_EVENTS: Event[] = [
  {
    event_id: 'EV_SYNTH_001_01',
    track_id: 'TRK_SYNTH_001',
    timestamp: '2024-07-01T00:00:00Z',
    centroid_lat: 18.42,
    centroid_lon: 86.85,
    bbox: [16.42, 20.42, 84.85, 88.85],
    area_km2: 12500,
    max_intensity: 118.5,
    mean_intensity: 46.2,
  },
  {
    event_id: 'EV_SYNTH_001_02',
    track_id: 'TRK_SYNTH_001',
    timestamp: '2024-07-01T06:00:00Z',
    centroid_lat: 19.15,
    centroid_lon: 86.20,
    bbox: [17.15, 21.15, 84.20, 88.20],
    area_km2: 14800,
    max_intensity: 124.8,
    mean_intensity: 51.7,
  },
  {
    event_id: 'EV_SYNTH_002_01',
    track_id: 'TRK_SYNTH_002',
    timestamp: '2024-07-02T00:00:00Z',
    centroid_lat: 14.20,
    centroid_lon: 81.60,
    bbox: [12.20, 16.20, 79.60, 83.60],
    area_km2: 10400,
    max_intensity: 96.4,
    mean_intensity: 38.9,
  },
  {
    event_id: 'EV_SYNTH_002_02',
    track_id: 'TRK_SYNTH_002',
    timestamp: '2024-07-02T06:00:00Z',
    centroid_lat: 15.05,
    centroid_lon: 80.95,
    bbox: [13.05, 17.05, 78.95, 82.95],
    area_km2: 11900,
    max_intensity: 105.2,
    mean_intensity: 42.1,
  },
  {
    event_id: 'EV_SYNTH_003_01',
    track_id: 'TRK_SYNTH_003',
    timestamp: '2024-07-03T00:00:00Z',
    centroid_lat: 21.80,
    centroid_lon: 88.40,
    bbox: [19.80, 23.80, 86.40, 90.40],
    area_km2: 16200,
    max_intensity: 138.2,
    mean_intensity: 58.4,
  },
  {
    event_id: 'EV_SYNTH_003_02',
    track_id: 'TRK_SYNTH_003',
    timestamp: '2024-07-03T06:00:00Z',
    centroid_lat: 22.45,
    centroid_lon: 87.75,
    bbox: [20.45, 24.45, 85.75, 89.75],
    area_km2: 17500,
    max_intensity: 142.0,
    mean_intensity: 62.1,
  },
  {
    event_id: 'EV_SYNTH_004_01',
    track_id: 'TRK_SYNTH_004',
    timestamp: '2024-07-04T00:00:00Z',
    centroid_lat: 12.60,
    centroid_lon: 76.80,
    bbox: [10.60, 14.60, 74.80, 78.80],
    area_km2: 8900,
    max_intensity: 78.0,
    mean_intensity: 29.5,
  },
  {
    event_id: 'EV_SYNTH_004_02',
    track_id: 'TRK_SYNTH_004',
    timestamp: '2024-07-04T06:00:00Z',
    centroid_lat: 13.25,
    centroid_lon: 76.15,
    bbox: [11.25, 15.25, 74.15, 78.15],
    area_km2: 9600,
    max_intensity: 84.5,
    mean_intensity: 32.8,
  },
]

export const FALLBACK_TRACKS: Track[] = [
  {
    track_id: 'TRK_SYNTH_001',
    event_ids: ['EV_SYNTH_001_01', 'EV_SYNTH_001_02'],
    timestamps: ['2024-07-01T00:00:00Z', '2024-07-01T06:00:00Z'],
    centroids: [
      [18.42, 86.85],
      [19.15, 86.20],
    ],
    bboxes: [
      [16.42, 20.42, 84.85, 88.85],
      [17.15, 21.15, 84.20, 88.20],
    ],
    speed_kmh: 24.5,
    bearing_deg: 318.0,
  },
  {
    track_id: 'TRK_SYNTH_002',
    event_ids: ['EV_SYNTH_002_01', 'EV_SYNTH_002_02'],
    timestamps: ['2024-07-02T00:00:00Z', '2024-07-02T06:00:00Z'],
    centroids: [
      [14.20, 81.60],
      [15.05, 80.95],
    ],
    bboxes: [
      [12.20, 16.20, 79.60, 83.60],
      [13.05, 17.05, 78.95, 82.95],
    ],
    speed_kmh: 21.8,
    bearing_deg: 324.5,
  },
  {
    track_id: 'TRK_SYNTH_003',
    event_ids: ['EV_SYNTH_003_01', 'EV_SYNTH_003_02'],
    timestamps: ['2024-07-03T00:00:00Z', '2024-07-03T06:00:00Z'],
    centroids: [
      [21.80, 88.40],
      [22.45, 87.75],
    ],
    bboxes: [
      [19.80, 23.80, 86.40, 90.40],
      [20.45, 24.45, 85.75, 89.75],
    ],
    speed_kmh: 28.2,
    bearing_deg: 315.0,
  },
  {
    track_id: 'TRK_SYNTH_004',
    event_ids: ['EV_SYNTH_004_01', 'EV_SYNTH_004_02'],
    timestamps: ['2024-07-04T00:00:00Z', '2024-07-04T06:00:00Z'],
    centroids: [
      [12.60, 76.80],
      [13.25, 76.15],
    ],
    bboxes: [
      [10.60, 14.60, 74.80, 78.80],
      [11.25, 15.25, 74.15, 78.15],
    ],
    speed_kmh: 19.4,
    bearing_deg: 320.0,
  },
]

export function createFallbackRisk(event: Event, threshold: number = 70): RiskResponse {
  const [minLat, maxLat, minLon, maxLon] = event.bbox
  const centerLat = event.centroid_lat
  const centerLon = event.centroid_lon
  const dLat = (maxLat - minLat) * 0.35
  const dLon = (maxLon - minLon) * 0.35

  const ring = [
    [centerLon - dLon, centerLat - dLat],
    [centerLon + dLon, centerLat - dLat],
    [centerLon + dLon * 1.2, centerLat + dLat * 0.5],
    [centerLon, centerLat + dLat * 1.1],
    [centerLon - dLon * 1.1, centerLat + dLat * 0.6],
    [centerLon - dLon, centerLat - dLat],
  ]

  const geojson: GeoJSON.FeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        type: 'Feature',
        geometry: {
          type: 'Polygon',
          coordinates: [ring],
        },
        properties: {
          event_id: event.event_id,
          threshold,
          max_prob: 0.88,
          mean_prob: 0.64,
          area_km2: event.area_km2 * 0.65,
        },
      },
    ],
  }

  return {
    event_id: event.event_id,
    track_id: event.track_id,
    timestamp: event.timestamp,
    threshold,
    max_exceedance_probability: 0.88,
    mean_exceedance_probability: 0.64,
    affected_area_km2: event.area_km2 * 0.65,
    severity_index: Math.min(1.0, event.max_intensity / 150),
    uncertainty: 0.18,
    risk_score: Math.min(100, Math.round((event.max_intensity / 150) * 85 + 10)),
    geojson,
    source_mode: 'synthetic_fallback',
  }
}
