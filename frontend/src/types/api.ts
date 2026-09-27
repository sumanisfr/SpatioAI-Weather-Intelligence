export type ModelStatus = {
  available: boolean
  loaded: boolean
  path?: string | null
  error?: string | null
}

export type HealthResponse = {
  status: string
  service: string
  version: string
  models_loaded: boolean
  models: Record<string, ModelStatus>
}

export type Event = {
  event_id: string
  track_id?: string | null
  timestamp: string
  centroid_lat: number
  centroid_lon: number
  bbox: [number, number, number, number]
  area_km2: number
  max_intensity: number
  mean_intensity: number
}

export type EventListResponse = {
  events: Event[]
  count: number
  source_mode: string
}

export type Track = {
  track_id: string
  event_ids: string[]
  timestamps: string[]
  centroids: [number, number][]
  bboxes: number[][]
  speed_kmh?: number | null
  bearing_deg?: number | null
}

export type TrackListResponse = {
  tracks: Track[]
  count: number
  source_mode: string
}

export type RiskResponse = {
  event_id: string
  track_id?: string | null
  timestamp: string
  threshold: number
  max_exceedance_probability: number
  mean_exceedance_probability: number
  affected_area_km2: number
  severity_index: number
  uncertainty: number
  risk_score: number
  geojson: GeoJSON.FeatureCollection
  source_mode: string
}

export type ApiError = Error & { status?: number }
