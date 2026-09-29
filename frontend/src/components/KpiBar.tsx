import type { KeyboardEvent } from 'react'
import { WindIcon, AlertTriangleIcon, CloudRainIcon, LayersIcon, TargetIcon } from './Icons'
import type { Event, Track } from '../types/api'

type KpiBarProps = {
  events: Event[]
  tracks: Track[]
  selectedEvent: Event | null
  isExtremeFiltered?: boolean
  onToggleExtremeFilter?: () => void
  onSelectPeakEvent?: () => void
  onTrackClick?: () => void
  onResetFootprint?: () => void
}

export function KpiBar({
  events,
  tracks,
  selectedEvent,
  isExtremeFiltered = false,
  onToggleExtremeFilter,
  onSelectPeakEvent,
  onTrackClick,
  onResetFootprint,
}: KpiBarProps) {
  // Peak anomaly across the entire domain
  const peakEvent = events.reduce<Event | null>((peak, ev) => {
    if (!peak || ev.max_intensity > peak.max_intensity) return ev
    return peak
  }, null)

  const peakIntensity = peakEvent ? peakEvent.max_intensity : 0
  const totalAreaKm2 = events.reduce((sum, ev) => sum + ev.area_km2, 0)
  const extremeCount = events.filter((ev) => ev.max_intensity >= 100).length
  const severeCount = events.length - extremeCount

  const handleKeyDown = (e: KeyboardEvent, action?: () => void) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      action?.()
    }
  }

  return (
    <section className="kpi-ribbon" aria-label="Key Performance Indicators">
      {/* Metric 1: Active Tracks (Clickable to switch to Trajectory analysis) */}
      <div
        className="kpi-cell interactive"
        onClick={onTrackClick}
        onKeyDown={(e) => handleKeyDown(e, onTrackClick)}
        role="button"
        tabIndex={0}
        title="Click to view GNN Storm Trajectories &amp; Timelines"
      >
        <div className="kpi-cell-icon text-sky-400">
          <WindIcon size={16} />
        </div>
        <div className="kpi-cell-content">
          <div className="kpi-cell-label">
            <span>TRACKED SYSTEMS</span>
            <span className="kpi-click-hint">View Tracks &rarr;</span>
          </div>
          <div className="kpi-cell-main">
            <span className="kpi-cell-num">{String(tracks.length).padStart(2, '0')}</span>
            <span className="kpi-cell-tag">Hungarian Kinematic</span>
          </div>
          <div className="kpi-cell-detail">
            Geodesic 6h Trajectory Continuity
          </div>
        </div>
      </div>

      {/* Metric 2: Detected Anomalies (Interactive filter toggle) */}
      <div
        className={`kpi-cell interactive ${isExtremeFiltered ? 'active-filter' : ''}`}
        onClick={onToggleExtremeFilter}
        onKeyDown={(e) => handleKeyDown(e, onToggleExtremeFilter)}
        role="button"
        tabIndex={0}
        title="Click to toggle filter: Show only extreme anomalies (>100 mm/h)"
      >
        <div className={`kpi-cell-icon ${isExtremeFiltered ? 'text-rose-400' : 'text-amber-400'}`}>
          <AlertTriangleIcon size={16} />
        </div>
        <div className="kpi-cell-content">
          <div className="kpi-cell-label">
            <span>EXTREME ANOMALIES</span>
            <span className={`kpi-filter-tag ${isExtremeFiltered ? 'active' : ''}`}>
              {isExtremeFiltered ? '● FILTER ON' : 'Filter Toggle'}
            </span>
          </div>
          <div className="kpi-cell-main">
            <span className="kpi-cell-num">{String(events.length).padStart(2, '0')}</span>
            <div className="kpi-pill-subinfo">
              <span className="kpi-badge-extreme">{extremeCount} Extreme</span>
              <span className="kpi-badge-severe">{severeCount} Severe</span>
            </div>
          </div>
          <div className="kpi-cell-detail">
            &gt;95th Climatology Percentile
          </div>
        </div>
      </div>

      {/* Metric 3: Peak Rain Rate (Interactive jump to peak event) */}
      <div
        className="kpi-cell interactive"
        onClick={() => {
          if (peakEvent && onSelectPeakEvent) onSelectPeakEvent()
        }}
        onKeyDown={(e) => handleKeyDown(e, () => peakEvent && onSelectPeakEvent?.())}
        role="button"
        tabIndex={0}
        title="Click to center map &amp; telemetry on domain maximum anomaly"
      >
        <div className="kpi-cell-icon text-amber-400">
          <CloudRainIcon size={16} />
        </div>
        <div className="kpi-cell-content">
          <div className="kpi-cell-label">
            <span>PEAK PRECIPITATION</span>
            <span className="kpi-click-hint">Jump to Peak</span>
          </div>
          <div className="kpi-cell-main">
            <span className="kpi-cell-num text-amber-400">{peakIntensity.toFixed(1)}</span>
            <span className="kpi-cell-unit">mm/h</span>
            {peakEvent && (
              <span className="kpi-focus-pill" title="Jump to peak anomaly">
                <TargetIcon size={10} className="inline mr-1 text-amber-400" />
                {peakEvent.event_id.replace('EV_SYNTH_', 'EV-')}
              </span>
            )}
          </div>
          <div className="kpi-cell-detail">
            {selectedEvent
              ? `Selected: ${selectedEvent.event_id.replace('EV_SYNTH_', 'EV-')} (${selectedEvent.max_intensity.toFixed(1)} mm/h)`
              : 'Domain Max Single-Cell Rate'}
          </div>
        </div>
      </div>

      {/* Metric 4: Aggregate Footprint Area (Interactive reset/overview) */}
      <div
        className="kpi-cell interactive"
        onClick={onResetFootprint}
        onKeyDown={(e) => handleKeyDown(e, onResetFootprint)}
        role="button"
        tabIndex={0}
        title="Click to reset map domain to All India"
      >
        <div className="kpi-cell-icon text-emerald-400">
          <LayersIcon size={16} />
        </div>
        <div className="kpi-cell-content">
          <div className="kpi-cell-label">
            <span>AGGREGATE FOOTPRINT</span>
            <span className="kpi-click-hint">All India Overview</span>
          </div>
          <div className="kpi-cell-main">
            <span className="kpi-cell-num text-emerald-300">
              {totalAreaKm2.toLocaleString('en-US', { maximumFractionDigits: 0 })}
            </span>
            <span className="kpi-cell-unit">km&sup2;</span>
          </div>
          <div className="kpi-cell-detail">
            Latitude-Weighted Spherical Grid Area
          </div>
        </div>
      </div>
    </section>
  )
}
