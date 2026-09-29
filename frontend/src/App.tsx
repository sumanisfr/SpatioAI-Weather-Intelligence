import { useState, useEffect } from 'react'
import { Header } from './components/Header'
import { NavigationSidebar, type NavItemKey } from './components/NavigationSidebar'
import { KpiBar } from './components/KpiBar'
import { MapPanel } from './components/MapPanel'
import { DetectedEventsPanel } from './components/DetectedEventsPanel'
import { EventDetail } from './components/EventDetail'
import { DownscalingPanel } from './components/DownscalingPanel'
import { EnsembleUncertaintyPanel } from './components/EnsembleUncertaintyPanel'
import { TrackTimeline } from './components/TrackTimeline'
import { ModelBenchmarkRow } from './components/ModelBenchmarkRow'
import { ModelRegistryCard } from './components/ModelRegistryCard'
import { LiveStationCard } from './components/LiveStationCard'
import { ErrorBoundary } from './components/ErrorBoundary'
import { useDashboardData } from './hooks/useDashboardData'
import { useNavigationState } from './hooks/useNavigationState'
import './App.css'

export function App() {
  const data = useDashboardData()
  const nav = useNavigationState(data.selectedId)

  // Navigation tab state
  const [activeNav, setActiveNav] = useState<NavItemKey>('dashboard')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [analysisToast, setAnalysisToast] = useState<string | null>(null)

  // Mobile responsiveness for sidebar
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 960) {
        setSidebarCollapsed(true)
      } else {
        setSidebarCollapsed(false)
      }
    }
    handleResize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  const navSelectedId = nav.selectedId
  const navSelectEvent = nav.selectEvent
  const dataSelectedId = data.selectedId
  const dataSetSelectedId = data.setSelectedId
  const dataEvents = data.events

  // Synchronize navigation selection with data selection
  useEffect(() => {
    if (navSelectedId && navSelectedId !== dataSelectedId) {
      dataSetSelectedId(navSelectedId)
    }
  }, [navSelectedId, dataSelectedId, dataSetSelectedId])

  useEffect(() => {
    if (dataSelectedId && !navSelectedId) {
      navSelectEvent(dataSelectedId)
    } else if (!navSelectedId && dataEvents.length > 0) {
      const def = dataEvents.find((e) => e.event_id.includes('003_02')) || dataEvents[0]
      navSelectEvent(def.event_id)
      dataSetSelectedId(def.event_id)
    }
  }, [dataSelectedId, dataEvents, navSelectedId, navSelectEvent, dataSetSelectedId])

  // Modals for sidebar actions
  const [showDataInputModal, setShowDataInputModal] = useState(false)
  const [showSettingsModal, setShowSettingsModal] = useState(false)

  const handleSelectEvent = (id: string) => {
    nav.selectEvent(id)
  }

  const handleSelectPeakEvent = () => {
    if (!data.events.length) return
    const peak = data.events.reduce(
      (max, ev) => (ev.max_intensity > max.max_intensity ? ev : max),
      data.events[0]
    )
    nav.selectEvent(peak.event_id)
  }

  const handleRunAnalysis = async () => {
    setAnalysisToast('Running neural meteorological anomaly pipeline...')
    await data.refresh()
    setTimeout(() => {
      setAnalysisToast('Analysis complete: 12km → 5km field super-resolved.')
      setTimeout(() => setAnalysisToast(null), 3000)
    }, 1200)
  }

  const handleSearch = (query: string) => {
    const q = query.toLowerCase().trim()
    if (!q) return
    if (q.includes('bengal') || q.includes('odisha') || q.includes('puri') || q.includes('vizag') || q.includes('kolkata')) {
      nav.setActiveBasin('bay_of_bengal')
      setAnalysisToast('Domain focused: Bay of Bengal (Odisha & Coastal Zone)')
    } else if (q.includes('penins') || q.includes('karnataka') || q.includes('tamil') || q.includes('chennai') || q.includes('bangalore')) {
      nav.setActiveBasin('peninsular')
      setAnalysisToast('Domain focused: Peninsular India Basin')
    } else if (q.includes('ghat') || q.includes('mumbai') || q.includes('maharashtra') || q.includes('kerala') || q.includes('goa')) {
      nav.setActiveBasin('western_ghats')
      setAnalysisToast('Domain focused: Western Ghats Orographic Zone')
    } else {
      nav.setActiveBasin('subcontinent')
      setAnalysisToast(`Domain focused: All-India Regional Grid for "${query}"`)
    }
  }

  const handleNavSelect = (key: NavItemKey) => {
    setActiveNav(key)
    const mainEl = document.querySelector('.spatio-main-content')
    if (key === 'events') {
      setActiveNav('dashboard')
      const el = document.querySelector('.detected-events-panel') as HTMLElement | null
      if (el && mainEl) {
        const top = Math.max(0, el.offsetTop - 16)
        mainEl.scrollTo({ top, behavior: 'smooth' })
      }
      setAnalysisToast('Viewing Detected Extreme Events')
      setTimeout(() => setAnalysisToast(null), 2500)
    } else if (key === 'tracking') {
      setActiveNav('dashboard')
      const el = document.querySelector('.timeline-panel-card') as HTMLElement | null
      if (el && mainEl) {
        const top = Math.max(0, el.offsetTop - 16)
        mainEl.scrollTo({ top, behavior: 'smooth' })
      }
      setAnalysisToast('Viewing Cyclone Trajectory and Intensity Timeline')
      setTimeout(() => setAnalysisToast(null), 2500)
    } else if (key === 'risk-maps') {
      setActiveNav('dashboard')
      const el = document.querySelector('.map-panel-container') as HTMLElement | null
      if (el && mainEl) {
        const top = Math.max(0, el.offsetTop - 16)
        mainEl.scrollTo({ top, behavior: 'smooth' })
      }
      setAnalysisToast('Risk Maps mode: Cyclone Doppler overlay active')
      setTimeout(() => setAnalysisToast(null), 2500)
    } else if (key === 'data-input') {
      setShowDataInputModal(true)
    } else if (key === 'settings') {
      setShowSettingsModal(true)
    }
  }

  return (
    <div className="spatio-app-container">
      {/* 1. Full-Width Top Header */}
      <Header
        health={data.health}
        loading={data.loading}
        sidebarCollapsed={sidebarCollapsed}
        onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
        onRefresh={() => void data.refresh()}
        activeBasin={nav.activeBasin}
        onSelectBasin={nav.setActiveBasin}
        onRunAnalysis={handleRunAnalysis}
        onSearch={handleSearch}
        onSelectVariable={(v) => {
          setAnalysisToast(`Variable switched to: ${v.toUpperCase()}`)
          setTimeout(() => setAnalysisToast(null), 2500)
        }}
        onSelectDateTime={(dt) => {
          setAnalysisToast(`Temporal step: ${dt}`)
          setTimeout(() => setAnalysisToast(null), 2500)
        }}
        canGoBack={nav.canGoBack}
        onGoBack={nav.goBack}
      />

      {/* Analysis Toast Notification */}
      {analysisToast && (
        <div className="analysis-toast-pill" role="status">
          <span className="toast-dot-green" />
          <span>{analysisToast}</span>
        </div>
      )}

      {/* Data Input Modal */}
      {showDataInputModal && (
        <div className="spatio-modal-backdrop" onClick={() => setShowDataInputModal(false)}>
          <div className="spatio-modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="spatio-modal-header">
              <h3 className="spatio-modal-title">Meteorological Data Input Pipeline</h3>
              <button
                type="button"
                className="spatio-modal-close"
                onClick={() => setShowDataInputModal(false)}
              >
                &times;
              </button>
            </div>
            <div className="spatio-modal-body">
              <p className="text-xs text-slate-300 mb-3">
                SpatioAI ingests multi-channel NWP grids and Doppler radar scans to run Spatio-Temporal Graph Neural Networks and Conditional Diffusion models.
              </p>
              <div className="schema-key-val-grid">
                <div className="schema-kv-row">
                  <span className="text-slate-400">Primary Ingest Format:</span>
                  <span className="text-sky-400 font-mono">CF-1.8 NetCDF4 / Zarr Tensors</span>
                </div>
                <div className="schema-kv-row">
                  <span className="text-slate-400">Input Grid Resolution:</span>
                  <span className="text-emerald-400 font-mono">12 km &times; 12 km (ERA5 / IMD)</span>
                </div>
                <div className="schema-kv-row">
                  <span className="text-slate-400">Target Super-Resolution:</span>
                  <span className="text-emerald-400 font-mono">5 km &times; 5 km (Diffusion Super-Res)</span>
                </div>
                <div className="schema-kv-row">
                  <span className="text-slate-400">Active Bounding Domain:</span>
                  <span className="text-amber-400 font-mono">8.0°N&ndash;37.0°N, 68.0°E&ndash;97.5°E</span>
                </div>
                <div className="schema-kv-row">
                  <span className="text-slate-400">Telemetry Ingest Interval:</span>
                  <span className="text-slate-200 font-mono">6-Hour Synoptic Cycles</span>
                </div>
              </div>
            </div>
            <div className="spatio-modal-footer">
              <button
                type="button"
                className="dock-tab-btn active text-xs"
                onClick={() => setShowDataInputModal(false)}
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Settings Modal */}
      {showSettingsModal && (
        <div className="spatio-modal-backdrop" onClick={() => setShowSettingsModal(false)}>
          <div className="spatio-modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="spatio-modal-header">
              <h3 className="spatio-modal-title">System Settings &amp; Thresholds</h3>
              <button
                type="button"
                className="spatio-modal-close"
                onClick={() => setShowSettingsModal(false)}
              >
                &times;
              </button>
            </div>
            <div className="spatio-modal-body">
              <div className="mb-4">
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Extreme Risk Threshold: {data.riskThreshold} mm
                </label>
                <input
                  type="range"
                  min="20"
                  max="120"
                  step="5"
                  value={data.riskThreshold}
                  onChange={(e) => data.setRiskThreshold(Number(e.target.value))}
                  className="w-full accent-sky-500 cursor-pointer"
                />
                <span className="text-[10px] text-slate-400">
                  Precipitation values exceeding this threshold are flagged as extreme anomalies.
                </span>
              </div>
              <div className="mb-3">
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Ensemble Confidence Bounds
                </label>
                <select className="filter-select-minimal w-full">
                  <option value="95">95% Confidence Interval (Operational Default)</option>
                  <option value="90">90% Confidence Interval</option>
                  <option value="99">99% Confidence Interval (Extreme Safety)</option>
                </select>
              </div>
            </div>
            <div className="spatio-modal-footer">
              <button
                type="button"
                className="dock-tab-btn active text-xs"
                onClick={() => {
                  setShowSettingsModal(false)
                  setAnalysisToast('Settings saved successfully.')
                  setTimeout(() => setAnalysisToast(null), 2500)
                }}
              >
                Apply &amp; Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 2. Workspace Body: Left Sidebar + Main Content */}
      <div className="spatio-workspace-row">
        {/* Left Fixed Navigation Sidebar */}
        <aside
          className={`spatio-sidebar-column ${sidebarCollapsed ? 'collapsed' : ''}`}
          aria-label="Navigation Sidebar"
        >
          <NavigationSidebar
            activeItem={activeNav}
            onSelectItem={handleNavSelect}
            health={data.health}
            onCloseMobile={() => {
              if (window.innerWidth < 960) setSidebarCollapsed(true)
            }}
          />
        </aside>

        {/* Mobile Backdrop Overlay */}
        {!sidebarCollapsed && (
          <div
            className="spatio-sidebar-mobile-backdrop"
            onClick={() => setSidebarCollapsed(true)}
            aria-hidden="true"
          />
        )}

        {/* Main Dashboard Content Area */}
        <main className="spatio-main-content" role="main">
          {/* If Model Status tab is clicked, show Model Registry */}
          {activeNav === 'model-status' && (
            <div className="tab-standalone-view">
              <div className="flex justify-between items-center mb-3">
                <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
                  Model Status &amp; Inference Checkpoints
                </h2>
                <button
                  type="button"
                  className="dock-tab-btn active text-xs"
                  onClick={() => setActiveNav('dashboard')}
                >
                  &larr; Return to Dashboard
                </button>
              </div>
              <ModelRegistryCard health={data.health} />
            </div>
          )}

          {/* If Forecast Analysis tab is clicked, show Station Ground Truth */}
          {activeNav === 'forecast-analysis' && (
            <div className="tab-standalone-view">
              <div className="flex justify-between items-center mb-3">
                <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
                  Forecast Telemetry &amp; Ground Truth Validation
                </h2>
                <button
                  type="button"
                  className="dock-tab-btn active text-xs"
                  onClick={() => setActiveNav('dashboard')}
                >
                  &larr; Return to Dashboard
                </button>
              </div>
              <LiveStationCard event={data.selectedEvent} />
            </div>
          )}

          {/* Primary View: Dashboard strictly matching reference screenshot */}
          {activeNav !== 'model-status' && activeNav !== 'forecast-analysis' && (
            <>
              {/* Row 1: KPI Summary Cards (5 cards) */}
              <KpiBar
                events={data.events}
                tracks={data.tracks}
                selectedEvent={data.selectedEvent}
                isExtremeFiltered={nav.extremeOnly}
                onToggleExtremeFilter={() => nav.setExtremeOnly((prev) => !prev)}
                onSelectPeakEvent={handleSelectPeakEvent}
                onResetFootprint={() => {
                  nav.setActiveBasin('bay_of_bengal')
                  nav.setExtremeOnly(false)
                }}
              />

              {/* Row 2: Middle Section (Map + Detected Events + Event Details) */}
              <div className="middle-dashboard-grid">
                {/* Column 1: Extreme Weather Events Map */}
                <div className="grid-col-map">
                  <ErrorBoundary fallbackTitle="Extreme Weather Events Map">
                    <MapPanel
                      allEvents={data.events}
                      event={data.selectedEvent}
                      track={data.selectedTrack}
                      risk={data.risk}
                      onSelectEvent={handleSelectEvent}
                      activeBasin={nav.activeBasin}
                    />
                  </ErrorBoundary>
                </div>

                {/* Column 2: Detected Events Panel */}
                <div className="grid-col-events">
                  <ErrorBoundary fallbackTitle="Detected Events">
                    <DetectedEventsPanel
                      events={data.events}
                      selectedId={data.selectedId}
                      onSelectEvent={handleSelectEvent}
                    />
                  </ErrorBoundary>
                </div>

                {/* Column 3: Event Details Panel */}
                <div className="grid-col-details">
                  <ErrorBoundary fallbackTitle="Event Details">
                    <EventDetail
                      event={data.selectedEvent}
                      risk={data.risk}
                    />
                  </ErrorBoundary>
                </div>
              </div>

              {/* Row 3: Analytics Row 1 (3 Panels) */}
              <div className="analytics-panels-grid">
                {/* Panel 1: High-Resolution Downscaling */}
                <div className="analytics-col">
                  <ErrorBoundary fallbackTitle="High-Resolution Downscaling">
                    <DownscalingPanel />
                  </ErrorBoundary>
                </div>

                {/* Panel 2: Ensemble Uncertainty */}
                <div className="analytics-col">
                  <ErrorBoundary fallbackTitle="Ensemble Uncertainty">
                    <EnsembleUncertaintyPanel
                      currentThreshold={data.riskThreshold}
                      onThresholdChange={data.setRiskThreshold}
                    />
                  </ErrorBoundary>
                </div>

                {/* Panel 3: Track Intensity Timeline */}
                <div className="analytics-col">
                  <ErrorBoundary fallbackTitle="Track Intensity Timeline">
                    <TrackTimeline
                      track={data.selectedTrack}
                      events={data.events}
                      selectedEventId={data.selectedId}
                      onSelectEvent={handleSelectEvent}
                    />
                  </ErrorBoundary>
                </div>
              </div>

              {/* Row 4: Benchmark Performance Row 2 (3 Panels) */}
              <ErrorBoundary fallbackTitle="Model Benchmark Performance">
                <ModelBenchmarkRow />
              </ErrorBoundary>
            </>
          )}
        </main>
      </div>
    </div>
  )
}

export default App

