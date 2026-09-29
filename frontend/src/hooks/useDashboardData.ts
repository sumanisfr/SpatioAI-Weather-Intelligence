import { useEffect, useState, useCallback } from 'react'
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
  const [riskThreshold, setRiskThreshold] = useState<number>(0) // 0 = dynamic 60% of peak

  const [loading, setLoading] = useState<boolean>(true)
  const [riskLoading, setRiskLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [riskError, setRiskError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  // Initial load inside effect with no synchronous setState
  useEffect(() => {
    let ignore = false
    const init = async () => {
      try {
        const [healthRes, eventRes, trackRes] = await Promise.all([
          apiFetch<HealthResponse>('/health'),
          fetchEvents(),
          fetchTracks(),
        ])
        if (!ignore) {
          setHealth(healthRes)
          const evList = (eventRes as EventListResponse).events
          setEvents(evList)
          setTracks((trackRes as TrackListResponse).tracks)
          setSelectedId((curr) => curr ?? evList[0]?.event_id ?? null)
          setLastUpdated(new Date())
          setLoading(false)
        }
      } catch (err) {
        if (!ignore) {
          setError(err instanceof Error ? err.message : 'Unable to reach the SpatioAI backend')
          setLoading(false)
        }
      }
    }
    void init()
    return () => {
      ignore = true
    }
  }, [])

  // Manual refresh callback
  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [healthRes, eventRes, trackRes] = await Promise.all([
        apiFetch<HealthResponse>('/health'),
        fetchEvents(),
        fetchTracks(),
      ])
      setHealth(healthRes)
      const evList = (eventRes as EventListResponse).events
      setEvents(evList)
      setTracks((trackRes as TrackListResponse).tracks)
      setSelectedId((curr) => curr ?? evList[0]?.event_id ?? null)
      setLastUpdated(new Date())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to reach the SpatioAI backend')
    } finally {
      setLoading(false)
    }
  }, [])

  const selectedEvent = events.find((ev) => ev.event_id === selectedId) ?? null
  const selectedTrack = tracks.find((tr) => tr.track_id === selectedEvent?.track_id) ?? null

  // Risk evaluation when selected event or threshold changes
  useEffect(() => {
    if (!selectedEvent) return
    let isCancelled = false
    const controller = new AbortController()

    const executeAnalysis = async () => {
      const thresholdToUse = riskThreshold > 0
        ? riskThreshold
        : Math.max(25, Math.round(selectedEvent.max_intensity * 0.6))

      try {
        const res = await analyzeRisk(
          selectedEvent.event_id,
          thresholdToUse,
          0.5,
          controller.signal
        )
        if (!isCancelled) {
          setRisk(res)
          setRiskError(null)
          setRiskLoading(false)
        }
      } catch (err) {
        if (!isCancelled && !controller.signal.aborted) {
          setRiskError(err instanceof Error ? err.message : 'Risk analysis currently unavailable')
          setRiskLoading(false)
        }
      }
    }

    void executeAnalysis()

    return () => {
      isCancelled = true
      controller.abort()
    }
  }, [selectedEvent, riskThreshold])

  return {
    health,
    events,
    tracks,
    selectedId,
    setSelectedId,
    selectedEvent,
    selectedTrack,
    risk,
    riskThreshold,
    setRiskThreshold,
    loading,
    riskLoading,
    error,
    riskError,
    lastUpdated,
    refresh,
  }
}
