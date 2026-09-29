import { useState, type KeyboardEvent } from 'react'
import {
  RadarIcon,
  SearchIcon,
  CloudRainIcon,
  CalendarIcon,
  MapPinIcon,
  PlayIcon,
  RefreshCwIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
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
  onRunAnalysis?: () => void
  onSearch?: (query: string) => void
  selectedVariable?: string
  onSelectVariable?: (variable: string) => void
  selectedDateTime?: string
  onSelectDateTime?: (dateTime: string) => void
  canGoBack?: boolean
  onGoBack?: () => void
}

export function Header({
  health: _health,
  loading,
  sidebarCollapsed,
  onToggleSidebar,
  onRefresh,
  activeBasin = 'bay_of_bengal',
  onSelectBasin,
  onRunAnalysis,
  onSearch,
  selectedVariable: controlledVariable,
  onSelectVariable,
  selectedDateTime: controlledDateTime,
  onSelectDateTime,
  canGoBack = false,
  onGoBack,
}: HeaderProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [internalVariable, setInternalVariable] = useState('rainfall')
  const [internalDateTime, setInternalDateTime] = useState('2026-08-18 12:00 UTC')

  const selectedVariable = controlledVariable ?? internalVariable
  const selectedDateTime = controlledDateTime ?? internalDateTime

  const handleVariableChange = (val: string) => {
    setInternalVariable(val)
    onSelectVariable?.(val)
  }

  const handleDateTimeChange = (val: string) => {
    setInternalDateTime(val)
    onSelectDateTime?.(val)
  }

  const regions = [
    { id: 'bay_of_bengal', label: 'Bay of Bengal' },
    { id: 'subcontinent', label: 'All India' },
    { id: 'peninsular', label: 'Peninsular' },
    { id: 'western_ghats', label: 'Western Ghats' },
  ]

  const variables = [
    { id: 'rainfall', label: 'Rainfall' },
    { id: 'wind', label: 'Wind Velocity' },
    { id: 'temp', label: 'Surface Temp' },
    { id: 'pressure', label: 'MSL Pressure' },
  ]

  const dateTimes = [
    '2026-08-18 12:00 UTC',
    '2026-08-18 06:00 UTC',
    '2026-08-17 18:00 UTC',
    '2026-08-17 12:00 UTC',
  ]

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSearch?.(searchQuery)
  }

  const handleKeyDown = (e: KeyboardEvent, action?: () => void) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      action?.()
    }
  }

  return (
    <header className="site-header" role="banner">
      {/* Left: Brand Logo & Title */}
      <div className="header-brand-section">
        {onGoBack && canGoBack && (
          <button
            className="header-back-btn active"
            onClick={onGoBack}
            title="Return to previous selection/view"
            aria-label="Go Back"
            type="button"
          >
            <ChevronLeftIcon size={15} />
          </button>
        )}

        <button
          className="sidebar-rail-btn"
          onClick={onToggleSidebar}
          title={sidebarCollapsed ? 'Expand Navigation Sidebar' : 'Collapse Navigation Sidebar'}
          aria-label="Toggle Navigation Sidebar"
          type="button"
        >
          {sidebarCollapsed ? <ChevronRightIcon size={16} /> : <ChevronLeftIcon size={16} />}
        </button>

        <div
          className="brand-lockup interactive"
          onClick={() => onSelectBasin?.('bay_of_bengal')}
          onKeyDown={(e) => handleKeyDown(e, () => onSelectBasin?.('bay_of_bengal'))}
          role="button"
          tabIndex={0}
          title="SpatioAI Meteorological Anomaly Tracking"
        >
          <div className="brand-mark-circle">
            <RadarIcon size={20} className="brand-mark-icon" />
          </div>
          <div className="brand-meta">
            <h1 className="brand-name">SpatioAI</h1>
            <span className="brand-tagline">
              Meteorological Anomaly Tracking &bull; 12km &rarr; 5km
            </span>
          </div>
        </div>
      </div>

      {/* Center: Search + Selectors Row */}
      <div className="header-center-controls">
        {/* Location Search Bar */}
        <form className="header-search-form" onSubmit={handleSearchSubmit}>
          <SearchIcon size={15} className="header-search-icon" />
          <input
            type="text"
            className="header-search-input"
            placeholder="Search location (e.g., Odisha, Bay of Bengal)"
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value)
              onSearch?.(e.target.value)
            }}
            aria-label="Search location"
          />
        </form>

        {/* Variable Selector Dropdown */}
        <div className="header-dropdown-wrap">
          <CloudRainIcon size={14} className="dropdown-prefix-icon text-sky-400" />
          <select
            className="header-select"
            value={selectedVariable}
            onChange={(e) => handleVariableChange(e.target.value)}
            aria-label="Select Weather Variable"
          >
            {variables.map((v) => (
              <option key={v.id} value={v.id}>
                {v.label}
              </option>
            ))}
          </select>
        </div>

        {/* Date/Time Selector Dropdown */}
        <div className="header-dropdown-wrap">
          <CalendarIcon size={14} className="dropdown-prefix-icon text-slate-400" />
          <select
            className="header-select"
            value={selectedDateTime}
            onChange={(e) => handleDateTimeChange(e.target.value)}
            aria-label="Select Date and Time"
          >
            {dateTimes.map((dt) => (
              <option key={dt} value={dt}>
                {dt}
              </option>
            ))}
          </select>
        </div>

        {/* Region Selector Dropdown */}
        <div className="header-dropdown-wrap">
          <MapPinIcon size={14} className="dropdown-prefix-icon text-emerald-400" />
          <select
            className="header-select"
            value={activeBasin}
            onChange={(e) => onSelectBasin?.(e.target.value)}
            aria-label="Select Meteorological Domain Region"
          >
            {regions.map((r) => (
              <option key={r.id} value={r.id}>
                {r.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Right: Primary Run Analysis Button & Utilities */}
      <div className="header-actions-section">
        <button
          className={`primary-run-analysis-btn ${loading ? 'loading' : ''}`}
          onClick={onRunAnalysis || onRefresh}
          disabled={loading}
          type="button"
          title="Execute SpatioAI Neural Anomaly Tracking Pipeline"
        >
          <PlayIcon size={13} className="run-icon" />
          <span>{loading ? 'Analyzing...' : 'Run Analysis'}</span>
        </button>

        <button
          className={`header-utility-btn ${loading ? 'spinning' : ''}`}
          onClick={onRefresh}
          disabled={loading}
          title="Refresh Telemetry Pipeline"
          aria-label="Refresh Telemetry"
          type="button"
        >
          <RefreshCwIcon size={15} />
        </button>
      </div>
    </header>
  )
}
