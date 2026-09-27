import { useEffect, useState } from 'react'
import { apiFetch } from '../api/client'
import { fetchEvents } from '../api/events'
import { fetchTracks } from '../api/tracking'
import { analyzeRisk } from '../api/risk'
import type { Event, EventListResponse, HealthResponse, RiskResponse, Track, TrackListResponse } from '../types/api'

export function useDashboardData() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [events, setEvents] = useState<Event[]>([])
  const [tracks, setTracks] = useState<Track[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [risk, setRisk] = useState<RiskResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [riskLoading, setRiskLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [riskError, setRiskError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [healthResponse, eventResponse, trackResponse] = await Promise.all([
        apiFetch<HealthResponse>('/health'),
        fetchEvents(),
        fetchTracks(),
      ])
      setHealth(healthResponse)
      setEvents((eventResponse as EventListResponse).events)
      setTracks((trackResponse as TrackListResponse).tracks)
      setSelectedId((current) => current ?? eventResponse.events[0]?.event_id ?? null)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to reach the backend')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void load() }, [])

  const selectedEvent = events.find((event) => event.event_id === selectedId) ?? null
  const selectedTrack = tracks.find((track) => track.track_id === selectedEvent?.track_id) ?? null

  useEffect(() => {
    if (!selectedEvent) return
    const controller = new AbortController()
    setRiskLoading(true)
    setRiskError(null)
    void analyzeRisk(selectedEvent.event_id, Math.max(25, Math.round(selectedEvent.max_intensity * 0.6)), 0.5, controller.signal)
      .then(setRisk)
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) setRiskError(cause instanceof Error ? cause.message : 'Risk analysis unavailable')
      })
      .finally(() => { if (!controller.signal.aborted) setRiskLoading(false) })
    return () => controller.abort()
  }, [selectedEvent])

  return { health, events, tracks, selectedEvent, selectedTrack, selectedId, setSelectedId, risk, loading, riskLoading, error, riskError, refresh: load }
}
