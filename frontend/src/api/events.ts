import { apiFetch } from './client'
import type { EventListResponse, Event } from '../types/api'

export function fetchEvents(signal?: AbortSignal) {
  return apiFetch<EventListResponse>('/api/v1/events', { signal })
}

export function fetchEvent(eventId: string) {
  return apiFetch<Event>(`/api/v1/events/${encodeURIComponent(eventId)}`)
}
