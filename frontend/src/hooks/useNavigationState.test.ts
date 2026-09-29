import { describe, expect, it } from 'vitest'
import { parseUrlState, buildNavigationUrl } from './useNavigationState'

describe('useNavigationState URL Utilities', () => {
  it('parses empty search query with default meteorological values', () => {
    const parsed = parseUrlState('')
    expect(parsed.eventId).toBeNull()
    expect(parsed.tab).toBe('all')
    expect(parsed.basin).toBe('subcontinent')
    expect(parsed.extremeOnly).toBe(false)
    expect(parsed.trackFilter).toBeNull()
    expect(parsed.mobileView).toBe('map')
  })

  it('correctly parses full URL parameters for deep-linking', () => {
    const query = '?event=EV_SYNTH_003_02&tab=telemetry&basin=bay_of_bengal&extreme=true&track=TRK-003&view=telemetry'
    const parsed = parseUrlState(query)

    expect(parsed.eventId).toBe('EV_SYNTH_003_02')
    expect(parsed.tab).toBe('telemetry')
    expect(parsed.basin).toBe('bay_of_bengal')
    expect(parsed.extremeOnly).toBe(true)
    expect(parsed.trackFilter).toBe('TRK-003')
    expect(parsed.mobileView).toBe('telemetry')
  })

  it('safely falls back when unknown tab or view parameters are provided', () => {
    const query = '?event=EV_SYNTH_001_01&tab=invalid_tab&view=invalid_view'
    const parsed = parseUrlState(query)

    expect(parsed.eventId).toBe('EV_SYNTH_001_01')
    expect(parsed.tab).toBe('all')
    expect(parsed.mobileView).toBe('map')
  })

  it('builds navigation URL with active filters correctly', () => {
    const url = buildNavigationUrl({
      eventId: 'EV_SYNTH_002_01',
      tab: 'risk',
      basin: 'peninsular',
      extremeOnly: true,
      trackFilter: 'TRK-002',
    })

    expect(url).toContain('event=EV_SYNTH_002_01')
    expect(url).toContain('tab=risk')
    expect(url).toContain('basin=peninsular')
    expect(url).toContain('extreme=true')
    expect(url).toContain('track=TRK-002')
  })

  it('omits default tab and basin in built URL for clean browser addresses', () => {
    const url = buildNavigationUrl({
      eventId: 'EV_SYNTH_001_01',
      tab: 'all',
      basin: 'subcontinent',
      extremeOnly: false,
    })

    expect(url).toBe('/?event=EV_SYNTH_001_01')
    expect(url).not.toContain('tab=')
    expect(url).not.toContain('basin=')
  })
})
