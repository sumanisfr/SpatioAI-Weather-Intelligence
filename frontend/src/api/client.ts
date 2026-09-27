import type { ApiError } from '../types/api'

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? 'http://localhost:8000'

export async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), 15000)
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, { ...options, signal: options.signal ?? controller.signal })
    const body = await response.json().catch(() => ({})) as { detail?: string }
    if (!response.ok) {
      const error = new Error(body.detail ?? `Request failed with status ${response.status}`) as ApiError
      error.status = response.status
      throw error
    }
    return body as T
  } finally {
    window.clearTimeout(timeout)
  }
}
