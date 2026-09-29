import { useState, useMemo } from 'react'
import { SearchIcon, FilterIcon, SlidersIcon, MapPinIcon, CloudRainIcon, LayersIcon } from './Icons'
import type { Event } from '../types/api'

type SidebarProps = {
  events: Event[]
  selectedId: string | null
  onSelectEvent: (eventId: string) => void
  loading: boolean
  error: string | null
  extremeOnly: boolean
  setExtremeOnly: (val: boolean | ((curr: boolean) => boolean)) => void
  trackFilter: string | null
  setTrackFilter: (val: string | null) => void
}

type SortField = 'intensity' | 'area' | 'time'

export function Sidebar({
  events,
  selectedId,
  onSelectEvent,
  loading,
  error,
  extremeOnly,
  setExtremeOnly,
  trackFilter,
  setTrackFilter,
}: SidebarProps) {
  const [query, setQuery] = useState('')
  const [sortBy, setSortBy] = useState<SortField>('intensity')

  const uniqueTracks = useMemo(() => {
    return Array.from(new Set(events.map((e) => e.track_id).filter(Boolean))) as string[]
  }, [events])

  const filteredEvents = useMemo(() => {
    const q = query.trim().toLowerCase()
    return events
      .filter((ev) => {
        if (trackFilter && ev.track_id !== trackFilter) return false
        if (extremeOnly && ev.max_intensity < 100) return false
        if (!q) return true
        return (
          ev.event_id.toLowerCase().includes(q) ||
          (ev.track_id && ev.track_id.toLowerCase().includes(q)) ||
          ev.timestamp.toLowerCase().includes(q) ||
          ev.centroid_lat.toFixed(2).includes(q) ||
          ev.centroid_lon.toFixed(2).includes(q)
        )
      })
      .sort((a, b) => {
        if (sortBy === 'intensity') return b.max_intensity - a.max_intensity
        if (sortBy === 'area') return b.area_km2 - a.area_km2
        return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      })
  }, [events, query, trackFilter, extremeOnly, sortBy])

  const formatTimestamp = (iso: string) => {
    try {
      const d = new Date(iso)
      const datePart = d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })
      const timePart = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })
      return `${datePart}, ${timePart} UTC`
    } catch {
      return iso
    }
  }

  return (
    <aside className="sidebar-container" aria-label="Detected Events Monitor">
      {/* Sidebar Header */}
      <div className="sidebar-top-bar">
        <div className="sidebar-title-group">
          <span className="sidebar-eyebrow">ANOMALY DETECTOR</span>
          <h2 className="sidebar-title">Candidate Events</h2>
        </div>
        <div className="sidebar-counter">
          <span className="counter-current">{filteredEvents.length}</span>
          <span className="counter-total">/{events.length}</span>
        </div>
      </div>

      {/* Quick Search */}
      <div className="sidebar-search-wrap">
        <SearchIcon size={14} className="search-icon-svg" />
        <input
          type="text"
          placeholder="Filter code, track, or coords..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Filter events"
          className="sidebar-search-input"
        />
        {query ? (
          <button
            className="search-clear-btn"
            onClick={() => setQuery('')}
            title="Clear filter"
            type="button"
          >
            &times;
          </button>
        ) : (
          <kbd className="search-kbd-hint">/</kbd>
        )}
      </div>

      {/* Filter Tabs */}
      <div className="sidebar-filter-row" role="group" aria-label="Filter events by category">
        <button
          className={`filter-pill-btn ${!trackFilter && !extremeOnly ? 'active' : ''}`}
          onClick={() => {
            setTrackFilter(null)
            setExtremeOnly(false)
          }}
          type="button"
        >
          All
        </button>

        <button
          className={`filter-pill-btn ${extremeOnly ? 'active' : ''}`}
          onClick={() => setExtremeOnly(!extremeOnly)}
          title="Filter events with max precipitation >= 100 mm/h"
          type="button"
        >
          &gt;100 mm/h
        </button>

        {uniqueTracks.map((trk) => (
          <button
            key={trk}
            className={`filter-pill-btn ${trackFilter === trk ? 'active' : ''}`}
            onClick={() => setTrackFilter(trackFilter === trk ? null : trk)}
            type="button"
          >
            {trk.replace('TRK_SYNTH_', 'TRK-')}
          </button>
        ))}
      </div>

      {/* Sort Selector Bar */}
      <div className="sidebar-sort-row">
        <span className="sort-hint-label">
          <SlidersIcon size={11} className="inline mr-1 text-slate-500" />
          Sort by:
        </span>
        <div className="sort-pill-group">
          <button
            className={`sort-pill ${sortBy === 'intensity' ? 'active' : ''}`}
            onClick={() => setSortBy('intensity')}
            type="button"
          >
            Peak Rain
          </button>
          <button
            className={`sort-pill ${sortBy === 'area' ? 'active' : ''}`}
            onClick={() => setSortBy('area')}
            type="button"
          >
            Footprint
          </button>
          <button
            className={`sort-pill ${sortBy === 'time' ? 'active' : ''}`}
            onClick={() => setSortBy('time')}
            type="button"
          >
            Time
          </button>
        </div>
      </div>

      {/* Event Cards Scrollable Feed */}
      <div className="sidebar-events-feed" role="listbox" aria-label="Detected events list">
        {loading && (
          <div className="sidebar-status-placeholder">
            <span className="loading-spinner" />
            <span className="text-xs text-slate-400">Loading candidate events...</span>
          </div>
        )}

        {error && (
          <div className="sidebar-status-placeholder error">
            <FilterIcon size={20} className="text-rose-400" />
            <span className="text-xs text-rose-300">{error}</span>
          </div>
        )}

        {!loading && !error && filteredEvents.length === 0 && (
          <div className="sidebar-status-placeholder empty">
            <span className="text-xs text-slate-400">No events match active criteria.</span>
            <button
              className="sidebar-reset-btn"
              onClick={() => {
                setQuery('')
                setTrackFilter(null)
                setExtremeOnly(false)
              }}
              type="button"
            >
              Reset Filters
            </button>
          </div>
        )}

        {!loading &&
          !error &&
          filteredEvents.map((ev) => {
            const isSelected = selectedId === ev.event_id
            const isExtreme = ev.max_intensity >= 110
            const isHeavy = ev.max_intensity >= 80 && ev.max_intensity < 110

            return (
              <div
                key={ev.event_id}
                className={`event-list-item ${isSelected ? 'selected' : ''}`}
                onClick={() => onSelectEvent(ev.event_id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    onSelectEvent(ev.event_id)
                  }
                }}
                role="option"
                aria-selected={isSelected}
                tabIndex={0}
              >
                {/* Top: Event Code, Track, and Timestamp */}
                <div className="item-top-row">
                  <div className="item-code-group">
                    <span
                      className={`item-severity-dot ${
                        isExtreme ? 'extreme' : isHeavy ? 'heavy' : 'moderate'
                      }`}
                    />
                    <span className="item-code-name">
                      {ev.event_id.replace('EV_SYNTH_', 'EV-')}
                    </span>
                    <span className="item-track-pill">
                      {ev.track_id ? ev.track_id.replace('TRK_SYNTH_', 'TRK-') : 'Untracked'}
                    </span>
                  </div>

                  <span className="item-timestamp">
                    {formatTimestamp(ev.timestamp)}
                  </span>
                </div>

                {/* Middle: Coordinates & Geographic Extent */}
                <div className="item-coords-row">
                  <span className="item-coord-text">
                    <MapPinIcon size={11} className="inline mr-1 text-slate-500" />
                    {ev.centroid_lat.toFixed(2)}&deg;N, {ev.centroid_lon.toFixed(2)}&deg;E
                  </span>
                  <span className="item-category-tag">
                    {isExtreme ? 'Catastrophic' : isHeavy ? 'Severe' : 'Moderate'}
                  </span>
                </div>

                {/* Bottom: Telemetry Metrics */}
                <div className="item-metrics-row">
                  <div className="item-metric-pair">
                    <span className="metric-pair-label">
                      <CloudRainIcon size={10} className="inline mr-1 text-slate-400" />
                      Peak Rate
                    </span>
                    <span
                      className={`metric-pair-val ${
                        isExtreme ? 'text-rose-400' : isHeavy ? 'text-amber-400' : 'text-sky-400'
                      }`}
                    >
                      {ev.max_intensity.toFixed(1)} <small>mm/h</small>
                    </span>
                  </div>

                  <div className="item-metric-pair right">
                    <span className="metric-pair-label">
                      <LayersIcon size={10} className="inline mr-1 text-slate-400" />
                      Footprint
                    </span>
                    <span className="metric-pair-val area">
                      {ev.area_km2.toLocaleString('en-US', { maximumFractionDigits: 0 })}{' '}
                      <small>km&sup2;</small>
                    </span>
                  </div>
                </div>

                {/* Micro Intensity Meter */}
                <div className="item-meter-track">
                  <div
                    className={`item-meter-fill ${
                      isExtreme ? 'extreme' : isHeavy ? 'heavy' : 'moderate'
                    }`}
                    style={{ width: `${Math.min(100, (ev.max_intensity / 150) * 100)}%` }}
                  />
                </div>
              </div>
            )
          })}
      </div>
    </aside>
  )
}
