import type { KeyboardEvent } from 'react'
import {
  RadarIcon,
  RefreshCwIcon,
  InfoIcon,
  ExternalLinkIcon,
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
  onOpenInfo: () => void
  activeBasin?: string
  onSelectBasin?: (basinId: string) => void
}

export function Header({
  health,
  loading,
  sidebarCollapsed,
  onToggleSidebar,
  onRefresh,
  onOpenInfo,
  activeBasin = 'subcontinent',
  onSelectBasin,
}: HeaderProps) {
  const isOnline = Boolean(health && health.status === 'ok')

  const basins = [
    { id: 'subcontinent', label: 'All India' },
    { id: 'bay_of_bengal', label: 'Bay of Bengal' },
    { id: 'peninsular', label: 'Peninsular' },
  ]

  const apiDocsUrl = `http://${typeof window !== 'undefined' && window.location.hostname ? window.location.hostname : '127.0.0.1'}:8000/docs`

  const handleKeyDown = (e: KeyboardEvent, action?: () => void) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      action?.()
    }
  }

  return (
    <header className="site-header" role="banner">
      {/* Left: Brand Identity & Sidebar Toggle */}
      <div className="header-brand-section">
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
              <span className="brand-version-pill">v{health?.version ?? '0.9.0'}</span>
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
        {/* Environment Badge (Interactive to open architecture info) */}
        <button
          className="env-pill interactive"
          onClick={onOpenInfo}
          title="Click to review Research Sandbox Architecture &amp; Methodology"
          type="button"
        >
          <span className="env-dot" />
          <span>Research Sandbox</span>
        </button>

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
            className="header-action-button"
            onClick={onOpenInfo}
            title="Scientific Methodology &amp; System Architecture"
            type="button"
          >
            <InfoIcon size={14} />
            <span>Architecture</span>
          </button>

          <a
            className="header-action-button"
            href={apiDocsUrl}
            target="_blank"
            rel="noreferrer"
            title="Open Interactive FastAPI Swagger Documentation"
          >
            <ExternalLinkIcon size={14} />
            <span>API Docs</span>
          </a>

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
