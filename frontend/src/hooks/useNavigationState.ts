import { useState, useEffect, useCallback } from 'react'

export type AnalysisTab = 'all' | 'telemetry' | 'risk' | 'models'
export type MobileView = 'map' | 'events' | 'telemetry' | 'risk' | 'models'

export interface NavigationParams {
  eventId: string | null
  tab: AnalysisTab
  basin: string
  extremeOnly: boolean
  trackFilter: string | null
  mobileView: MobileView
}

export function buildNavigationUrl(params: Partial<NavigationParams>, basePath = '/'): string {
  const searchParams = new URLSearchParams()
  if (params.eventId) searchParams.set('event', params.eventId)
  if (params.tab && params.tab !== 'all') searchParams.set('tab', params.tab)
  if (params.basin && params.basin !== 'subcontinent') searchParams.set('basin', params.basin)
  if (params.extremeOnly) searchParams.set('extreme', 'true')
  if (params.trackFilter) searchParams.set('track', params.trackFilter)
  if (params.mobileView && params.mobileView !== 'map') searchParams.set('view', params.mobileView)

  const q = searchParams.toString()
  return q ? `${basePath}?${q}` : basePath
}

export function parseUrlState(searchString?: string): NavigationParams {
  if (typeof window === 'undefined' && searchString === undefined) {
    return {
      eventId: null,
      tab: 'all',
      basin: 'subcontinent',
      extremeOnly: false,
      trackFilter: null,
      mobileView: 'map',
    }
  }

  const search = searchString !== undefined ? searchString : window.location.search
  const params = new URLSearchParams(search)
  const eventId = params.get('event') || null
  const tabParam = params.get('tab') as AnalysisTab | null
  const tab = tabParam && ['all', 'telemetry', 'risk', 'models'].includes(tabParam) ? tabParam : 'all'
  const basin = params.get('basin') || 'subcontinent'
  const extremeOnly = params.get('extreme') === 'true'
  const trackFilter = params.get('track') || null
  const viewParam = params.get('view') as MobileView | null
  const mobileView = viewParam && ['map', 'events', 'telemetry', 'risk', 'models'].includes(viewParam) ? viewParam : 'map'

  return { eventId, tab, basin, extremeOnly, trackFilter, mobileView }
}

export function useNavigationState(defaultEventId: string | null) {
  const [selectedId, setSelectedIdState] = useState<string | null>(() => parseUrlState().eventId || defaultEventId)
  const [analysisTab, setAnalysisTabState] = useState<AnalysisTab>(() => parseUrlState().tab)
  const [activeBasin, setActiveBasinState] = useState<string>(() => parseUrlState().basin)
  const [extremeOnly, setExtremeOnlyState] = useState<boolean>(() => parseUrlState().extremeOnly)
  const [trackFilter, setTrackFilterState] = useState<string | null>(() => parseUrlState().trackFilter)
  const [mobileView, setMobileViewState] = useState<MobileView>(() => parseUrlState().mobileView)

  // Track if we can go back in history
  const [canGoBack, setCanGoBack] = useState<boolean>(false)

  // Effective selected ID falls back to defaultEventId if state is null
  const currentSelectedId = selectedId ?? defaultEventId

  // Sync state to URL and history stack
  const syncToUrl = useCallback((
    nextEventId: string | null,
    nextTab: AnalysisTab,
    nextBasin: string,
    nextExtreme: boolean,
    nextTrack: string | null,
    nextView: MobileView,
    pushHistory = true
  ) => {
    if (typeof window === 'undefined') return

    const params = new URLSearchParams()
    if (nextEventId) params.set('event', nextEventId)
    if (nextTab && nextTab !== 'all') params.set('tab', nextTab)
    if (nextBasin && nextBasin !== 'subcontinent') params.set('basin', nextBasin)
    if (nextExtreme) params.set('extreme', 'true')
    if (nextTrack) params.set('track', nextTrack)
    if (nextView && nextView !== 'map') params.set('view', nextView)

    const search = params.toString() ? `?${params.toString()}` : ''
    const newUrl = `${window.location.pathname}${search}`

    if (pushHistory) {
      window.history.pushState(
        {
          eventId: nextEventId,
          tab: nextTab,
          basin: nextBasin,
          extremeOnly: nextExtreme,
          trackFilter: nextTrack,
          mobileView: nextView,
        },
        '',
        newUrl
      )
      setCanGoBack(true)
    } else {
      window.history.replaceState(
        {
          eventId: nextEventId,
          tab: nextTab,
          basin: nextBasin,
          extremeOnly: nextExtreme,
          trackFilter: nextTrack,
          mobileView: nextView,
        },
        '',
        newUrl
      )
    }
  }, [])

  // Listen to popstate (browser back and forward buttons)
  useEffect(() => {
    const handlePopState = (e: PopStateEvent) => {
      const state = e.state as NavigationParams | null
      const parsed = state || parseUrlState()

      setSelectedIdState(parsed.eventId)
      setAnalysisTabState(parsed.tab)
      setActiveBasinState(parsed.basin)
      setExtremeOnlyState(parsed.extremeOnly)
      setTrackFilterState(parsed.trackFilter)
      setMobileViewState(parsed.mobileView)
    }

    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  // Public setters that push to history
  const selectEvent = useCallback((eventId: string | null) => {
    setSelectedIdState(eventId)
    syncToUrl(eventId, analysisTab, activeBasin, extremeOnly, trackFilter, mobileView, true)
  }, [analysisTab, activeBasin, extremeOnly, trackFilter, mobileView, syncToUrl])

  const setAnalysisTab = useCallback((tab: AnalysisTab) => {
    setAnalysisTabState(tab)
    syncToUrl(selectedId, tab, activeBasin, extremeOnly, trackFilter, mobileView, true)
  }, [selectedId, activeBasin, extremeOnly, trackFilter, mobileView, syncToUrl])

  const setActiveBasin = useCallback((basin: string) => {
    setActiveBasinState(basin)
    syncToUrl(selectedId, analysisTab, basin, extremeOnly, trackFilter, mobileView, true)
  }, [selectedId, analysisTab, extremeOnly, trackFilter, mobileView, syncToUrl])

  const setExtremeOnly = useCallback((val: boolean | ((curr: boolean) => boolean)) => {
    setExtremeOnlyState((curr) => {
      const next = typeof val === 'function' ? val(curr) : val
      syncToUrl(selectedId, analysisTab, activeBasin, next, trackFilter, mobileView, true)
      return next
    })
  }, [selectedId, analysisTab, activeBasin, trackFilter, mobileView, syncToUrl])

  const setTrackFilter = useCallback((track: string | null) => {
    setTrackFilterState(track)
    syncToUrl(selectedId, analysisTab, activeBasin, extremeOnly, track, mobileView, true)
  }, [selectedId, analysisTab, activeBasin, extremeOnly, mobileView, syncToUrl])

  const setMobileView = useCallback((view: MobileView) => {
    setMobileViewState(view)
    syncToUrl(selectedId, analysisTab, activeBasin, extremeOnly, trackFilter, view, true)
  }, [selectedId, analysisTab, activeBasin, extremeOnly, trackFilter, syncToUrl])

  // In-app Back Action
  const goBack = useCallback(() => {
    if (typeof window === 'undefined') return
    if (window.history.length > 1) {
      window.history.back()
    } else {
      // Default reset if at root
      setSelectedIdState(defaultEventId)
      setAnalysisTabState('all')
      setActiveBasinState('subcontinent')
      setMobileViewState('map')
      syncToUrl(defaultEventId, 'all', 'subcontinent', false, null, 'map', false)
    }
  }, [defaultEventId, syncToUrl])

  return {
    selectedId: currentSelectedId,
    selectEvent,
    analysisTab,
    setAnalysisTab,
    activeBasin,
    setActiveBasin,
    extremeOnly,
    setExtremeOnly,
    trackFilter,
    setTrackFilter,
    mobileView,
    setMobileView,
    canGoBack,
    goBack,
  }
}
