import { AlertTriangleIcon, MapPinIcon, CloudRainIcon, LayersIcon, ActivityIcon } from './Icons'
import type { Event } from '../types/api'

type EventDetailProps = {
  event: Event | null
}

export function EventDetail({ event }: EventDetailProps) {
  if (!event) {
    return (
      <section className="card-panel event-detail-card" aria-label="Event Telemetry Inspection">
        <div className="panel-eyebrow">CLUSTER TELEMETRY</div>
        <h3 className="panel-title">Anomaly Properties</h3>
        <div className="state-placeholder empty">
          <AlertTriangleIcon size={20} className="text-slate-500" />
          <span className="text-xs text-slate-400">Select an event from the spatial monitor to inspect detailed telemetry.</span>
        </div>
      </section>
    )
  }

  const isCatastrophic = event.max_intensity >= 110
  const isSevere = event.max_intensity >= 80 && event.max_intensity < 110

  const severityText = isCatastrophic
    ? 'Catastrophic Cloudburst (>P99)'
    : isSevere
    ? 'Severe Convective Cell (>P95)'
    : 'Intense Mesoscale Precipitation'

  const concentrationRatio = event.mean_intensity > 0
    ? (event.max_intensity / event.mean_intensity).toFixed(1)
    : '1.0'

  return (
    <section className="card-panel event-detail-card" aria-label="Selected Event Telemetry">
      <div className="card-header-row">
        <div>
          <span className="panel-eyebrow">CLUSTER SEGMENTATION</span>
          <h3 className="panel-title">{event.event_id.replace('EV_SYNTH_', 'EV-')}</h3>
        </div>
        <div className="severity-badge-wrap">
          <span className={`severity-tag ${isCatastrophic ? 'catastrophic' : isSevere ? 'severe' : 'moderate'}`}>
            {severityText}
          </span>
        </div>
      </div>

      {/* Primary Metrics Grid */}
      <div className="detail-stat-grid">
        <div className="stat-box">
          <span className="stat-title">
            <CloudRainIcon size={12} className="text-amber-400 inline mr-1" />
            Peak Rate
          </span>
          <div className="stat-number-wrap">
            <span className="stat-number text-amber-400">{event.max_intensity.toFixed(1)}</span>
            <span className="stat-unit">mm/h</span>
          </div>
          <span className="stat-desc">Maximum single-cell flux</span>
        </div>

        <div className="stat-box">
          <span className="stat-title">
            <CloudRainIcon size={12} className="text-sky-400 inline mr-1" />
            Mean Rate
          </span>
          <div className="stat-number-wrap">
            <span className="stat-number">{event.mean_intensity.toFixed(1)}</span>
            <span className="stat-unit">mm/h</span>
          </div>
          <span className="stat-desc">Spatial cluster mean</span>
        </div>

        <div className="stat-box">
          <span className="stat-title">
            <LayersIcon size={12} className="text-indigo-400 inline mr-1" />
            Footprint
          </span>
          <div className="stat-number-wrap">
            <span className="stat-number">{event.area_km2.toLocaleString('en-US', { maximumFractionDigits: 0 })}</span>
            <span className="stat-unit">km&sup2;</span>
          </div>
          <span className="stat-desc">Latitude-weighted area</span>
        </div>

        <div className="stat-box">
          <span className="stat-title">
            <MapPinIcon size={12} className="text-teal-400 inline mr-1" />
            Centroid
          </span>
          <div className="stat-number-wrap">
            <span className="stat-number text-sm">{event.centroid_lat.toFixed(2)}&deg;N</span>
            <span className="stat-number text-sm ml-1 text-slate-400">{event.centroid_lon.toFixed(2)}&deg;E</span>
          </div>
          <span className="stat-desc">Intensity-weighted coords</span>
        </div>
      </div>

      {/* Geospatial Extent Bounds & Convective Indices */}
      <div className="extent-box">
        <div className="extent-header">
          <span>Spatial Bounding Box Extent (EPSG:4326)</span>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Peak/Mean Ratio: <strong className="text-sky-400 font-mono">{concentrationRatio}&times;</strong></span>
            <code className="text-xs text-slate-400 font-mono">Track: {event.track_id ? event.track_id.replace('TRK_SYNTH_', 'TRK-') : 'Untracked'}</code>
          </div>
        </div>
        <div className="extent-coords-grid">
          <div><span>North (Max Lat):</span> <strong>{event.bbox[1].toFixed(2)}&deg;N</strong></div>
          <div><span>South (Min Lat):</span> <strong>{event.bbox[0].toFixed(2)}&deg;N</strong></div>
          <div><span>East (Max Lon):</span> <strong>{event.bbox[3].toFixed(2)}&deg;E</strong></div>
          <div><span>West (Min Lon):</span> <strong>{event.bbox[2].toFixed(2)}&deg;E</strong></div>
        </div>
      </div>

      {/* Observation Time Details */}
      <div className="event-detail-footer">
        <div className="text-xs text-slate-400 flex items-center gap-1.5">
          <ActivityIcon size={12} className="text-sky-400" />
          <span>Forecast Observation: <strong className="font-mono text-slate-300">{new Date(event.timestamp).toUTCString().slice(5, 22)}</strong></span>
        </div>
      </div>
    </section>
  )
}
