import { apiFetch } from './client'
import type { TrackListResponse, Track } from '../types/api'

export function fetchTracks(signal?: AbortSignal) {
  return apiFetch<TrackListResponse>('/api/v1/tracks', { signal })
}

export function fetchTrack(trackId: string) {
  return apiFetch<Track>(`/api/v1/tracks/${encodeURIComponent(trackId)}`)
}
