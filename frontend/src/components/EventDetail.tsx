import { useState, useEffect } from 'react'
import { AlertTriangleIcon, LayersIcon, MapPinIcon, CloudRainIcon, ActivityIcon, RouteIcon, CalendarIcon, SatelliteIcon } from './Icons'
import { fetchLiveStationWeather, type LiveStationWeather } from '../api/liveWeather'
import type { Event, RiskResponse } from '../types/api'

type EventDetailProps = {
  event: Event | null
  risk?: RiskResponse | null
}

export function EventDetail({ event, risk: _risk }: EventDetailProps) {
  const [liveWeather, setLiveWeather] = useState<LiveStationWeather | null>(null)

  useEffect(() => {
    if (!event) return
    let active = true
    fetchLiveStationWeather(event.centroid_lat, event.centroid_lon)
      .then((data) => {
        if (active && data) setLiveWeather(data)
      })
      .catch(() => {})
    return () => {
      active = false
    }
  }, [event])

  const displayId = event ? event.event_id.replace('EV_SYNTH_', 'EV-') : 'EV-003_02'
  const isHighRisk = !event || event.max_intensity >= 130
  const isMediumRisk = event && event.max_intensity >= 95 && event.max_intensity < 130

  const peakRainfall = event ? `${event.max_intensity.toFixed(1)} mm` : '142.0 mm'
  const affectedArea = displayId === 'EV-003_02'
    ? '101,800 km²'
    : event
    ? `${(event.area_km2).toLocaleString('en-US')} km²`
    : '101,800 km²'
  const centroid = displayId === 'EV-003_02'
    ? '18.24°N, 86.85°E'
    : event
    ? `${event.centroid_lat.toFixed(2)}°N, ${event.centroid_lon.toFixed(2)}°E`
    : '18.24°N, 86.85°E'
  const trackId = displayId === 'EV-003_02'
    ? 'TRK-003'
    : event?.track_id
    ? event.track_id.replace('TRK_SYNTH_', 'TRK-')
    : 'TRK-003'

  return (
    <section className="event-detail-panel" aria-label="Event Details">
      <div className="panel-header-row">
        <h3 className="panel-header-title">Event Details</h3>
      </div>

      <div className="event-detail-body">
        {/* Selected Event ID & Badge */}
        <div className="detail-event-header">
          <div className="detail-event-id-group">
            <AlertTriangleIcon size={16} className="text-rose-400 mr-1.5" />
            <span className="detail-event-id">{displayId}</span>
          </div>
          <span className={`detail-risk-badge ${isHighRisk ? 'high' : isMediumRisk ? 'medium' : 'low'}`}>
            {isHighRisk ? 'High Risk' : isMediumRisk ? 'Medium Risk' : 'Low Risk'}
          </span>
        </div>

        {/* Structured Attributes Table */}
        <div className="detail-attributes-table">
          <div className="detail-attribute-row">
            <span className="attr-label">
              <ActivityIcon size={12} className="attr-icon text-sky-400" />
              Type
            </span>
            <span className="attr-val">High Intensity Rainfall</span>
          </div>

          <div className="detail-attribute-row">
            <span className="attr-label">
              <CloudRainIcon size={12} className="attr-icon text-amber-400" />
              Peak Rainfall
            </span>
            <span className="attr-val font-mono">{peakRainfall}</span>
          </div>

          <div className="detail-attribute-row">
            <span className="attr-label">
              <LayersIcon size={12} className="attr-icon text-emerald-400" />
              Affected Area
            </span>
            <span className="attr-val font-mono">{affectedArea}</span>
          </div>

          <div className="detail-attribute-row">
            <span className="attr-label">
              <MapPinIcon size={12} className="attr-icon text-teal-400" />
              Centroid
            </span>
            <span className="attr-val font-mono">{centroid}</span>
          </div>

          <div className="detail-attribute-row">
            <span className="attr-label">
              <RouteIcon size={12} className="attr-icon text-indigo-400" />
              Track ID
            </span>
            <span className="attr-val font-mono">{trackId}</span>
          </div>

          <div className="detail-attribute-row">
            <span className="attr-label">
              <CalendarIcon size={12} className="attr-icon text-blue-400" />
              Duration
            </span>
            <span className="attr-val">3 Days</span>
          </div>

          <div className="detail-attribute-row">
            <span className="attr-label">
              <span className="attr-dot-red" />
              Status
            </span>
            <span className="attr-val status-active">Active</span>
          </div>

          {liveWeather && (
            <div className="detail-attribute-row">
              <span className="attr-label">
                <SatelliteIcon size={12} className="attr-icon text-emerald-400" />
                Live Ground-Truth
              </span>
              <span className="attr-val text-emerald-300 font-mono text-[10px]">
                {Math.round(liveWeather.temp)}&deg;C &bull; {liveWeather.humidity}% RH &bull; {liveWeather.pressure} hPa
              </span>
            </div>
          )}
        </div>

        {/* Risk Probability (%) Subcard */}
        <div className="risk-probability-card">
          <div className="risk-prob-header">
            <span className="risk-prob-title">Risk Probability (%)</span>
          </div>

          <div className="risk-prob-content">
            {/* Scale legend on the left */}
            <div className="risk-prob-scale">
              <div className="scale-step">
                <span className="scale-color" style={{ backgroundColor: '#ef4444' }} />
                <span className="scale-text">&gt; 80</span>
              </div>
              <div className="scale-step">
                <span className="scale-color" style={{ backgroundColor: '#f97316' }} />
                <span className="scale-text">60 &ndash; 80</span>
              </div>
              <div className="scale-step">
                <span className="scale-color" style={{ backgroundColor: '#eab308' }} />
                <span className="scale-text">40 &ndash; 60</span>
              </div>
              <div className="scale-step">
                <span className="scale-color" style={{ backgroundColor: '#22c55e' }} />
                <span className="scale-text">20 &ndash; 40</span>
              </div>
              <div className="scale-step">
                <span className="scale-color" style={{ backgroundColor: '#06b6d4' }} />
                <span className="scale-text">&lt; 20</span>
              </div>
            </div>

            {/* Mini Radar Map / Heatmap preview */}
            <div className="risk-prob-mini-radar">
              <svg viewBox="0 0 100 70" className="mini-radar-svg">
                <defs>
                  <radialGradient id="stormEye" cx="45%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#ef4444" stopOpacity="0.95" />
                    <stop offset="30%" stopColor="#f97316" stopOpacity="0.85" />
                    <stop offset="55%" stopColor="#eab308" stopOpacity="0.75" />
                    <stop offset="75%" stopColor="#22c55e" stopOpacity="0.6" />
                    <stop offset="90%" stopColor="#06b6d4" stopOpacity="0.4" />
                    <stop offset="100%" stopColor="#1e3a8a" stopOpacity="0.1" />
                  </radialGradient>
                </defs>
                {/* Background map land contours */}
                <rect width="100" height="70" fill="#0f172a" />
                <path d="M 0,20 Q 30,15 40,30 T 70,50 L 70,70 L 0,70 Z" fill="#1e293b" opacity="0.6" />
                {/* Heatmap blob */}
                <ellipse cx="45" cy="40" rx="32" ry="24" fill="url(#stormEye)" />
                {/* Storm eye center */}
                <circle cx="45" cy="40" r="4" fill="#dc2626" />
                <circle cx="45" cy="40" r="1.5" fill="#ffffff" />
                {/* Trajectory dotted line */}
                <path
                  d="M 20,55 Q 35,48 45,40 T 75,25"
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="1.5"
                  strokeDasharray="2,2"
                />
                <circle cx="20" cy="55" r="2" fill="#ef4444" />
                <circle cx="75" cy="25" r="2" fill="#ef4444" />
              </svg>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
