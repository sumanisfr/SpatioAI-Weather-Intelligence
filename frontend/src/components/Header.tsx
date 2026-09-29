import type { KeyboardEvent } from 'react'
import {
  RadarIcon,
  RefreshCwIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CompassIcon,
} from './Icons'
import type { HealthResponse } from '../types/api'

type HeaderProps = {
  health: HealthResponse | null
  loading: boolean
  sidebarCollapsed: boolean
  onToggleSidebar: () => void
  onRefresh: () => void
  activeBasin?: string
  onSelectBasin?: (basinId: string) => void
  canGoBack?: boolean
  onGoBack?: () => void
}

export function Header({
  health,
  loading,
  sidebarCollapsed,
  onToggleSidebar,
  onRefresh,
  activeBasin = 'subcontinent',
  onSelectBasin,
  canGoBack = false,
  onGoBack,
}: HeaderProps) {
  const isOnline = Boolean(health && health.status === 'ok')

  const basins = [
    { id: 'subcontinent', label: 'All India' },
    { id: 'bay_of_bengal', label: 'Bay of Bengal' },
    { id: 'peninsular', label: 'Peninsular' },
    { id: 'western_ghats', label: 'Western Ghats' },
  ]

  const handleKeyDown = (e: KeyboardEvent, action?: () => void) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      action?.()
    }
  }

  return (
    <header className="site-header" role="banner">
      {/* Left: Brand Identity & Navigation Controls */}
      <div className="header-brand-section">
        {onGoBack && (
          <button
            className={`header-back-btn ${canGoBack ? 'active' : 'idle'}`}
            onClick={onGoBack}
            title={canGoBack ? 'Return to previous selection/view' : 'Reset to default basin & overview'}
            aria-label="Go Back to Previous State"
            type="button"
          >
            <ChevronLeftIcon size={15} />
            <span className="header-back-label">Back</span>
          </button>
        )}

        <button
          className="sidebar-rail-btn"
          onClick={onToggleSidebar}
          title={sidebarCollapsed ? 'Expand Event Monitor (Ctrl+B)' : 'Collapse Event Monitor (Ctrl+B)'}
          aria-label="Toggle Event Monitor Sidebar"
          type="button"
        >
          {sidebarCollapsed ? <ChevronRightIcon size={16} /> : <ChevronLeftIcon size={16} />}
        </button>

        <div
          className="brand-lockup interactive"
          onClick={() => onSelectBasin?.('subcontinent')}
          onKeyDown={(e) => handleKeyDown(e, () => onSelectBasin?.('subcontinent'))}
          role="button"
          tabIndex={0}
          title="Click to reset view to All India Subcontinent"
        >
          <div className="brand-mark">
            <RadarIcon size={18} className="brand-mark-icon" />
          </div>
          <div className="brand-meta">
            <div className="brand-heading">
              <span className="brand-name">SPATIO</span>
              <span className="brand-name-ai">AI</span>
            </div>
            <div className="brand-tagline">
              Meteorological Anomaly Tracking &bull; 12km &rarr; 5km
            </div>
          </div>
        </div>
      </div>

      {/* Center: Basin Navigation Presets */}
      <div className="header-nav-section">
        <div className="basin-segment" role="group" aria-label="Geographical domain presets">
          <span className="basin-segment-label">
            <CompassIcon size={12} className="text-slate-400" />
            Domain:
          </span>
          {basins.map((b) => (
            <button
              key={b.id}
              className={`basin-segment-btn ${activeBasin === b.id ? 'active' : ''}`}
              onClick={() => onSelectBasin?.(b.id)}
              type="button"
            >
              {b.label}
            </button>
          ))}
        </div>
      </div>

      {/* Right: Environment, Status, and Global Actions */}
      <div className="header-actions-section">
        {/* Environment Badge */}
        <div
          className="env-pill"
          title="SpatioAI Meteorological Research Sandbox"
        >
          <span className="env-dot" />
          <span>Research Sandbox</span>
        </div>

        {/* Backend Connectivity Status (Interactive to poll health) */}
        <button
          className={`system-status-indicator interactive ${isOnline ? 'online' : 'offline'}`}
          onClick={onRefresh}
          title={isOnline ? 'FastAPI Backend Online — Click to refresh telemetry' : 'Backend Connection Offline — Click to retry connection'}
          type="button"
        >
          <span className="status-ping-dot" />
          <span className="status-text">{isOnline ? 'Connected' : 'Offline'}</span>
        </button>

        {/* Action Controls */}
        <div className="action-button-group">
          <button
            className={`header-action-button icon-only ${loading ? 'loading' : ''}`}
            onClick={onRefresh}
            disabled={loading}
            title="Refresh pipeline telemetry"
            aria-label="Refresh Dashboard Telemetry"
            type="button"
          >
            <RefreshCwIcon size={14} className={loading ? 'animate-spin text-sky-400' : ''} />
          </button>
        </div>
      </div>
    </header>
  )
}
