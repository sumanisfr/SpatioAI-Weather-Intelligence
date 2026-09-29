import { useState } from 'react'
import { Header } from './components/Header'
import { KpiBar } from './components/KpiBar'
import { Sidebar } from './components/Sidebar'
import { MapPanel } from './components/MapPanel'
import { EventDetail } from './components/EventDetail'
import { TrackTimeline } from './components/TrackTimeline'
import { RiskPanel } from './components/RiskPanel'
import { LiveStationCard } from './components/LiveStationCard'
import { ErrorBoundary } from './components/ErrorBoundary'
import { RadarIcon, AlertTriangleIcon, ActivityIcon, ShieldAlertIcon } from './components/Icons'
import { useEffect } from 'react'
import { useDashboardData } from './hooks/useDashboardData'
import { useNavigationState, type AnalysisTab } from './hooks/useNavigationState'
import './App.css'

export function App() {
  const data = useDashboardData()
  const nav = useNavigationState(data.selectedId)

  // On mobile screens (<768px), sidebar starts collapsed to eliminate initial overlap
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    if (typeof window !== 'undefined') {
      return window.innerWidth < 768
    }
    return false
  })

  const navSelectedId = nav.selectedId
  const navSelectEvent = nav.selectEvent
  const dataSelectedId = data.selectedId
  const dataSetSelectedId = data.setSelectedId

  // Synchronize nav.selectedId with data.setSelectedId
  useEffect(() => {
    if (navSelectedId && navSelectedId !== dataSelectedId) {
      dataSetSelectedId(navSelectedId)
    }
  }, [navSelectedId, dataSelectedId, dataSetSelectedId])

  // Synchronize when data loads initial event if nav doesn't have one
  useEffect(() => {
    if (dataSelectedId && !navSelectedId) {
      navSelectEvent(dataSelectedId)
    }
  }, [dataSelectedId, navSelectedId, navSelectEvent])

  const handleSelectPeakEvent = () => {
    if (!data.events.length) return
    const peak = data.events.reduce((max, ev) => (ev.max_intensity > max.max_intensity ? ev : max), data.events[0])
    nav.selectEvent(peak.event_id)
  }

  const scrollToDiagnostics = (tab: AnalysisTab) => {
    nav.setAnalysisTab(tab)
    const el = document.querySelector('.analytics-dock-toolbar')
    el?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="dashboard-app-root">
      {/* Top Header with Back Navigation */}
      <Header
        health={data.health}
        loading={data.loading}
        sidebarCollapsed={sidebarCollapsed}
        onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
        onRefresh={() => void data.refresh()}
        activeBasin={nav.activeBasin}
        onSelectBasin={nav.setActiveBasin}
        canGoBack={nav.canGoBack || Boolean(nav.selectedId && data.events[0] && nav.selectedId !== data.events[0].event_id)}
        onGoBack={nav.goBack}
      />

      {/* KPI Metric Ribbon */}
      <KpiBar
        events={data.events}
        tracks={data.tracks}
        selectedEvent={data.selectedEvent}
        isExtremeFiltered={nav.extremeOnly}
        onToggleExtremeFilter={() => nav.setExtremeOnly((prev) => !prev)}
        onSelectPeakEvent={handleSelectPeakEvent}
        onTrackClick={() => scrollToDiagnostics('risk')}
        onResetFootprint={() => {
          nav.setActiveBasin('subcontinent')
          nav.setExtremeOnly(false)
          nav.setTrackFilter(null)
        }}
      />

      {/* Main Workspace Layout */}
      <div className="main-workspace-layout">
        {/* Left Collapsible Event Monitor Drawer */}
        <div className={`workspace-sidebar-pane ${sidebarCollapsed ? 'collapsed' : ''}`}>
          <ErrorBoundary fallbackTitle="Event Monitor Feed">
            <Sidebar
              events={data.events}
              selectedId={data.selectedId}
              onSelectEvent={(id) => {
                nav.selectEvent(id)
                // On mobile screens (<768px), auto-collapse drawer when user picks an event
                if (typeof window !== 'undefined' && window.innerWidth <= 768) {
                  setSidebarCollapsed(true)
                }
              }}
              loading={data.loading}
              error={data.error}
              extremeOnly={nav.extremeOnly}
              setExtremeOnly={nav.setExtremeOnly}
              trackFilter={nav.trackFilter}
              setTrackFilter={nav.setTrackFilter}
              onClose={() => setSidebarCollapsed(true)}
            />
          </ErrorBoundary>
        </div>

        {/* Mobile Backdrop Overlay */}
        {!sidebarCollapsed && (
          <div
            className="sidebar-mobile-backdrop"
            onClick={() => setSidebarCollapsed(true)}
            aria-hidden="true"
          />
        )}

        {/* Center / Right Analysis Console */}
        <main className="workspace-content-pane" role="main">
          {/* Geospatial Radar Intelligence Map */}
          <ErrorBoundary fallbackTitle="GIS Radar Workstation">
            <MapPanel
              allEvents={data.events}
              event={data.selectedEvent}
              track={data.selectedTrack}
              risk={data.risk}
              onSelectEvent={(id) => nav.selectEvent(id)}
              activeBasin={nav.activeBasin}
            />
          </ErrorBoundary>

          {/* Analytical Workspace Toolbar & Tabs */}
          <div className="analytics-dock-toolbar">
            <div className="dock-title-group">
              <span className="dock-eyebrow">METEOROLOGICAL INTELLIGENCE DOCK</span>
              <h3 className="dock-title">Diagnostic Telemetry, Risk &amp; Kinematics</h3>
            </div>

            <div className="dock-tabs-group" role="tablist" aria-label="Analysis section tabs">
              <button
                className={`dock-tab-btn ${nav.analysisTab === 'all' ? 'active' : ''}`}
                onClick={() => nav.setAnalysisTab('all')}
                role="tab"
                aria-selected={nav.analysisTab === 'all'}
                type="button"
              >
                Comprehensive View
              </button>
              <button
                className={`dock-tab-btn ${nav.analysisTab === 'telemetry' ? 'active' : ''}`}
                onClick={() => nav.setAnalysisTab('telemetry')}
                role="tab"
                aria-selected={nav.analysisTab === 'telemetry'}
                type="button"
              >
                Telemetry &amp; Ground-Truth
              </button>
              <button
                className={`dock-tab-btn ${nav.analysisTab === 'risk' ? 'active' : ''}`}
                onClick={() => nav.setAnalysisTab('risk')}
                role="tab"
                aria-selected={nav.analysisTab === 'risk'}
                type="button"
              >
                Uncertainty Risk &amp; Kinematics
              </button>
            </div>
          </div>

          {/* Analytical Panels in Balanced 2-Column Responsive Layout */}
          {(nav.analysisTab === 'all' || nav.analysisTab === 'telemetry') && (
            <div className="dock-grid-2col">
              <EventDetail event={data.selectedEvent} />
              <LiveStationCard event={data.selectedEvent} />
            </div>
          )}

          {(nav.analysisTab === 'all' || nav.analysisTab === 'risk') && (
            <div className="dock-grid-2col">
              <RiskPanel
                risk={data.risk}
                loading={data.riskLoading}
                error={data.riskError}
                currentThreshold={data.riskThreshold}
                onThresholdChange={data.setRiskThreshold}
                selectedEvent={data.selectedEvent}
              />
              <TrackTimeline
                track={data.selectedTrack}
                events={data.events}
                selectedEventId={data.selectedId}
              />
            </div>
          )}
        </main>
      </div>

      {/* Professional Engineering Footer */}
      <footer className="site-footer" role="contentinfo">
        <div className="footer-left-info">
          <strong>SpatioAI Meteorological Intelligence Console</strong> &bull; Indian Subcontinent Basin (8&deg;N&ndash;28&deg;N, 68&deg;E&ndash;94&deg;E)
        </div>
        <div className="footer-right-info">
          <span>FastAPI Backend &bull; Mode: <code>synthetic_experimental_demo</code> &bull; Grid: <code>12km &rarr; 5km</code></span>
        </div>
      </footer>

      {/* Mobile Bottom Navigation Bar (Visible on mobile screens <768px) */}
      <nav className="mobile-bottom-nav" aria-label="Mobile quick navigation">
        <button
          className={`mobile-nav-btn ${sidebarCollapsed && nav.mobileView === 'map' ? 'active' : ''}`}
          onClick={() => {
            setSidebarCollapsed(true)
            nav.setMobileView('map')
            window.scrollTo({ top: 0, behavior: 'smooth' })
          }}
          type="button"
        >
          <RadarIcon size={16} />
          <span>Radar Map</span>
        </button>
        <button
          className={`mobile-nav-btn ${!sidebarCollapsed ? 'active' : ''}`}
          onClick={() => {
            setSidebarCollapsed((prev) => !prev)
            nav.setMobileView('events')
          }}
          type="button"
        >
          <AlertTriangleIcon size={16} />
          <span>Anomalies</span>
        </button>
        <button
          className={`mobile-nav-btn ${nav.analysisTab === 'telemetry' && sidebarCollapsed ? 'active' : ''}`}
          onClick={() => {
            setSidebarCollapsed(true)
            nav.setMobileView('telemetry')
            scrollToDiagnostics('telemetry')
          }}
          type="button"
        >
          <ActivityIcon size={16} />
          <span>Telemetry</span>
        </button>
        <button
          className={`mobile-nav-btn ${nav.analysisTab === 'risk' && sidebarCollapsed ? 'active' : ''}`}
          onClick={() => {
            setSidebarCollapsed(true)
            nav.setMobileView('risk')
            scrollToDiagnostics('risk')
          }}
          type="button"
        >
          <ShieldAlertIcon size={16} />
          <span>Risk &amp; Tracks</span>
        </button>
      </nav>
    </div>
  )
}

export default App
