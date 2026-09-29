import { useState } from 'react'
import { Header } from './components/Header'
import { KpiBar } from './components/KpiBar'
import { Sidebar } from './components/Sidebar'
import { MapPanel } from './components/MapPanel'
import { EventDetail } from './components/EventDetail'
import { TrackTimeline } from './components/TrackTimeline'
import { RiskPanel } from './components/RiskPanel'
import { ModelRegistryCard } from './components/ModelRegistryCard'
import { DownscalingGate } from './components/DownscalingGate'
import { LiveStationCard } from './components/LiveStationCard'
import { InfoModal } from './components/InfoModal'
import { ErrorBoundary } from './components/ErrorBoundary'
import { useDashboardData } from './hooks/useDashboardData'
import './App.css'

type AnalysisTab = 'all' | 'telemetry' | 'risk' | 'models'

export function App() {
  const data = useDashboardData()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [infoModalOpen, setInfoModalOpen] = useState(false)
  const [extremeOnly, setExtremeOnly] = useState(false)
  const [trackFilter, setTrackFilter] = useState<string | null>(null)
  const [activeBasin, setActiveBasin] = useState<string>('subcontinent')
  const [analysisTab, setAnalysisTab] = useState<AnalysisTab>('all')

  const handleSelectPeakEvent = () => {
    if (!data.events.length) return
    const peak = data.events.reduce((max, ev) => (ev.max_intensity > max.max_intensity ? ev : max), data.events[0])
    data.setSelectedId(peak.event_id)
  }

  return (
    <div className="dashboard-app-root">
      {/* Ambient Gradient Orbs (decorative, pointer-events-none) */}
      <div style={{
        position: 'fixed', top: '35%', left: '50%', transform: 'translate(-50%,-50%)',
        width: '40vw', height: '40vw', borderRadius: '50%', pointerEvents: 'none', zIndex: 0,
        background: 'radial-gradient(ellipse at center, rgba(59,130,246,0.05) 0%, transparent 70%)',
      }} aria-hidden="true" />
      <div style={{
        position: 'fixed', top: '70%', left: '20%',
        width: '30vw', height: '30vw', borderRadius: '50%', pointerEvents: 'none', zIndex: 0,
        background: 'radial-gradient(ellipse at center, rgba(236,72,153,0.07) 0%, transparent 70%)',
        animation: 'float-orb 25s ease-in-out infinite',
      }} aria-hidden="true" />

      {/* Top Header */}
      <Header
        health={data.health}
        loading={data.loading}
        sidebarCollapsed={sidebarCollapsed}
        onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
        onRefresh={() => void data.refresh()}
        onOpenInfo={() => setInfoModalOpen(true)}
        activeBasin={activeBasin}
        onSelectBasin={setActiveBasin}
      />

      {/* KPI Metric Ribbon */}
      <KpiBar
        events={data.events}
        tracks={data.tracks}
        selectedEvent={data.selectedEvent}
        isExtremeFiltered={extremeOnly}
        onToggleExtremeFilter={() => setExtremeOnly((prev) => !prev)}
        onSelectPeakEvent={handleSelectPeakEvent}
        onTrackClick={() => setAnalysisTab('risk')}
        onResetFootprint={() => {
          setActiveBasin('subcontinent')
          setExtremeOnly(false)
          setTrackFilter(null)
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
              onSelectEvent={data.setSelectedId}
              loading={data.loading}
              error={data.error}
              extremeOnly={extremeOnly}
              setExtremeOnly={setExtremeOnly}
              trackFilter={trackFilter}
              setTrackFilter={setTrackFilter}
            />
          </ErrorBoundary>
        </div>

        {/* Center / Right Analysis Console */}
        <main className="workspace-content-pane" role="main">
          {/* Geospatial Radar Intelligence Map */}
          <ErrorBoundary fallbackTitle="GIS Radar Workstation">
            <MapPanel
              allEvents={data.events}
              event={data.selectedEvent}
              track={data.selectedTrack}
              risk={data.risk}
              onSelectEvent={data.setSelectedId}
              activeBasin={activeBasin}
            />
          </ErrorBoundary>

          {/* Analytical Workspace Toolbar & Tabs */}
          <div className="analytics-dock-toolbar">
            <div className="dock-title-group">
              <span className="dock-eyebrow">METEOROLOGICAL INTELLIGENCE DOCK</span>
              <h3 className="dock-title">Diagnostic Telemetry &amp; Models</h3>
            </div>

            <div className="dock-tabs-group" role="tablist" aria-label="Analysis section tabs">
              <button
                className={`dock-tab-btn ${analysisTab === 'all' ? 'active' : ''}`}
                onClick={() => setAnalysisTab('all')}
                role="tab"
                aria-selected={analysisTab === 'all'}
                type="button"
              >
                Comprehensive View
              </button>
              <button
                className={`dock-tab-btn ${analysisTab === 'telemetry' ? 'active' : ''}`}
                onClick={() => setAnalysisTab('telemetry')}
                role="tab"
                aria-selected={analysisTab === 'telemetry'}
                type="button"
              >
                Telemetry &amp; Ground-Truth
              </button>
              <button
                className={`dock-tab-btn ${analysisTab === 'risk' ? 'active' : ''}`}
                onClick={() => setAnalysisTab('risk')}
                role="tab"
                aria-selected={analysisTab === 'risk'}
                type="button"
              >
                Risk &amp; Trajectory
              </button>
              <button
                className={`dock-tab-btn ${analysisTab === 'models' ? 'active' : ''}`}
                onClick={() => setAnalysisTab('models')}
                role="tab"
                aria-selected={analysisTab === 'models'}
                type="button"
              >
                Model Architecture
              </button>
            </div>
          </div>

          {/* Analytical Panels in Balanced 2-Column Responsive Layout */}
          {(analysisTab === 'all' || analysisTab === 'telemetry') && (
            <div className="dock-grid-2col">
              <EventDetail event={data.selectedEvent} />
              <LiveStationCard event={data.selectedEvent} />
            </div>
          )}

          {(analysisTab === 'all' || analysisTab === 'risk') && (
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

          {(analysisTab === 'all' || analysisTab === 'models') && (
            <div className="dock-grid-2col">
              <ModelRegistryCard health={data.health} />
              <DownscalingGate />
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

      {/* Scientific Scope Modal */}
      <InfoModal
        isOpen={infoModalOpen}
        onClose={() => setInfoModalOpen(false)}
      />
    </div>
  )
}

export default App
