import { afterEach, describe, expect, it, vi } from 'vitest'
import { apiFetch } from './client'

afterEach(() => vi.restoreAllMocks())

describe('apiFetch', () => {
  it('surfaces backend detail and status for failed requests', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: 'model unavailable' }), { status: 503 })))
    await expect(apiFetch('/health')).rejects.toMatchObject({ message: 'model unavailable', status: 503 })
  })
})
