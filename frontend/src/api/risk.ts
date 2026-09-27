import { apiFetch } from './client'
import type { RiskResponse } from '../types/api'

export function analyzeRisk(eventId: string, threshold: number, probabilityThreshold = 0.5, signal?: AbortSignal) {
  return apiFetch<RiskResponse>('/api/v1/risk/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ event_id: eventId, threshold, probability_threshold: probabilityThreshold, ensemble_samples: 5, seed: 42 }),
    signal,
  })
}
