import { useEffect, useState, useRef, useMemo } from 'react'
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Polyline,
  Polygon,
  Marker,
  ImageOverlay,
  Tooltip,
  useMap,
} from 'react-leaflet'
import L from 'leaflet'
import {
  LayersIcon,
  RouteIcon,
  FlameIcon,
  SatelliteIcon,
  CrosshairIcon,
} from './Icons'
import type { Event, RiskResponse, Track } from '../types/api'
import { CYCLONE_BOUNDS, CYCLONE_RADAR_SVG_DATA_URL } from '../utils/cycloneRadarSvg'
import 'leaflet/dist/leaflet.css'

type MapPanelProps = {
  allEvents: Event[]
  event: Event | null
  track: Track | null
  risk: RiskResponse | null
  onSelectEvent: (eventId: string) => void
  activeBasin?: string
}

function ViewportManager({
  event,
  activeBasin,
}: {
  event: Event | null
  activeBasin?: string
}) {
  const map = useMap()
  const isInitializedRef = useRef(false)
  const prevEventIdRef = useRef<string | null>(null)
  const prevBasinRef = useRef<string | undefined>(activeBasin)

  useEffect(() => {
    if (!isInitializedRef.current) {
      isInitializedRef.current = true
      map.invalidateSize()
      map.setView([18.24, 86.85], 5)
    }
  }, [map])

  useEffect(() => {
    if (activeBasin && activeBasin !== prevBasinRef.current) {
      prevBasinRef.current = activeBasin
      if (activeBasin === 'bay_of_bengal') {
        map.flyTo([18.24, 86.85], 5, { duration: 0.8 })
      } else if (activeBasin === 'peninsular') {
        map.flyTo([14.0, 77.5], 6, { duration: 0.8 })
      } else if (activeBasin === 'western_ghats') {
        map.flyTo([18.5, 73.5], 6, { duration: 0.8 })
      } else {
        map.flyTo([20.0, 82.0], 5, { duration: 0.8 })
      }
    }
  }, [activeBasin, map])

  useEffect(() => {
    if (event && event.event_id !== prevEventIdRef.current) {
      prevEventIdRef.current = event.event_id
      map.flyTo([event.centroid_lat, event.centroid_lon], 5, { duration: 0.8 })
    }
  }, [event, map])

  return null
}

function ZoomButtons() {
  const map = useMap()
  const stopPropagation = (e: React.MouseEvent | React.TouchEvent) => {
    e.stopPropagation()
  }

  return (
    <div
      className="map-zoom-controls"
      onMouseDown={stopPropagation}
      onDoubleClick={stopPropagation}
      onTouchStart={stopPropagation}
    >
      <button
        type="button"
        className="map-zoom-btn"
        onClick={(e) => {
          e.stopPropagation()
          map.zoomIn()
        }}
        aria-label="Zoom in"
      >
        +
      </button>
      <button
        type="button"
        className="map-zoom-btn"
        onClick={(e) => {
          e.stopPropagation()
          map.zoomOut()
        }}
        aria-label="Zoom out"
      >
        &minus;
      </button>
    </div>
  )
}

// Trajectory coordinates matching reference image
const DEFAULT_TRAJECTORY: [number, number][] = [
  [15.2, 82.5],
  [16.8, 84.4],
  [18.24, 86.85], // Center / Eye of storm
  [19.8, 89.2],
  [21.2, 91.5],
]

// Meteorological uncertainty polygon swath along trajectory
const UNCERTAINTY_CONE: [number, number][] = [
  [14.5, 81.6],
  [15.9, 83.2],
  [17.3, 85.0],
  [19.5, 87.4],
  [21.8, 90.4],
  [22.8, 92.8],
  [21.2, 93.6],
  [19.0, 91.2],
  [17.5, 88.5],
  [16.2, 86.0],
  [14.8, 83.8],
  [13.8, 82.2],
]

export function MapPanel({
  allEvents,
  event,
  track: _track,
  risk: _risk,
  onSelectEvent,
  activeBasin = 'bay_of_bengal',
}: MapPanelProps) {
  const [showLayers, setShowLayers] = useState(false)
  const [showTrack, setShowTrack] = useState(true)
  const [showRisk, setShowRisk] = useState(true)
  const [showUncertainty, setShowUncertainty] = useState(true)
  const [tileMode, setTileMode] = useState<'satellite' | 'dark'>('satellite')
  const [showBorders, setShowBorders] = useState(true)

  const trajectory = DEFAULT_TRAJECTORY

  // Basemap tiles
  const tileUrl =
    tileMode === 'satellite'
      ? 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
      : 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}'

  // Pinned callout icon attached to storm location
  const calloutLat = event ? event.centroid_lat : 18.24
  const calloutLon = event ? event.centroid_lon : 86.85
  const eventIdDisplay = event ? event.event_id.replace('EV_SYNTH_', 'EV-') : 'EV-003_02'
  const eventIntensityDisplay = event ? event.max_intensity.toFixed(1) : '142.0'

  const eventCalloutIcon = useMemo(() => {
    return L.divIcon({
      className: 'leaflet-custom-callout-icon',
      html: `
        <div class="map-floating-event-callout leaflet-pinned">
          <div class="callout-header">
            <span class="callout-id">${eventIdDisplay}</span>
            <span class="callout-badge-risk">High Risk</span>
          </div>
          <div class="callout-type">Heavy Rainfall</div>
          <div class="callout-intensity">
            <span class="callout-dot-red"></span>
            <span>${eventIntensityDisplay} mm</span>
          </div>
          <div class="callout-time">18 Aug 2026, 12:00 UTC</div>
        </div>
      `,
      iconSize: [136, 72],
      iconAnchor: [68, 76],
    })
  }, [eventIdDisplay, eventIntensityDisplay])

  const stopToolbarPropagation = (e: React.MouseEvent | React.TouchEvent) => {
    e.stopPropagation()
  }

  return (
    <div className="map-panel-container" aria-label="Extreme Weather Events Map">
      {/* Header */}
      <div className="map-panel-header">
        <h3 className="map-panel-title">Extreme Weather Events Map</h3>
      </div>

      {/* Main Map Viewport */}
      <div className="map-leaflet-wrapper">
        <MapContainer
          center={[18.24, 86.85]}
          zoom={5}
          zoomControl={false}
          scrollWheelZoom={true}
          className="leaflet-dashboard-map"
        >
          {/* Satellite or Dark Base Layer */}
          <TileLayer
            attribution="&copy; Esri &mdash; Earthstar Geographics"
            url={tileUrl}
            maxZoom={16}
            minZoom={3}
          />

          {/* Reference Borders & Labels */}
          {showBorders && (
            <TileLayer
              attribution=""
              url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
              maxZoom={16}
            />
          )}

          <ViewportManager event={event} activeBasin={activeBasin} />

          {/* 1. Cyclone Radar Reflectivity Swirl (Geographically Pinned via Leaflet ImageOverlay) */}
          {showRisk && (
            <ImageOverlay
              url={CYCLONE_RADAR_SVG_DATA_URL}
              bounds={CYCLONE_BOUNDS}
              opacity={0.92}
              zIndex={400}
            />
          )}

          {/* 2. Ensemble Uncertainty Cone Swath */}
          {showUncertainty && (
            <Polygon
              positions={UNCERTAINTY_CONE}
              pathOptions={{
                color: '#38bdf8',
                fillColor: '#0284c7',
                fillOpacity: 0.18,
                weight: 1.5,
                dashArray: '5, 5',
              }}
            />
          )}

          {/* 3. Storm Trajectory Line with Waypoints */}
          {showTrack && (
            <>
              <Polyline
                positions={trajectory}
                pathOptions={{
                  color: '#ef4444',
                  weight: 3.5,
                  opacity: 0.95,
                }}
              />
              {trajectory.map((pos, idx) => (
                <CircleMarker
                  key={`pt-${idx}`}
                  center={pos}
                  radius={idx === 2 ? 6 : 4}
                  pathOptions={{
                    color: '#ffffff',
                    fillColor: idx === 2 ? '#ef4444' : '#ffffff',
                    fillOpacity: 1,
                    weight: 2,
                  }}
                />
              ))}
            </>
          )}

          {/* 4. Geographically Pinned Event Callout Card */}
          {showRisk && (
            <Marker
              position={[calloutLat + 1.1, calloutLon]}
              icon={eventCalloutIcon}
              interactive={false}
            />
          )}

          {/* 5. Candidate Weather Anomaly Clusters */}
          {allEvents.map((ev) => {
            const isSelected = event && ev.event_id === event.event_id
            if (isSelected) return null
            return (
              <CircleMarker
                key={ev.event_id}
                center={[ev.centroid_lat, ev.centroid_lon]}
                radius={5}
                pathOptions={{
                  color: '#ffffff',
                  fillColor: ev.max_intensity >= 120 ? '#ef4444' : '#f97316',
                  fillOpacity: 0.9,
                  weight: 1.5,
                }}
                eventHandlers={{
                  click: () => onSelectEvent(ev.event_id),
                }}
              >
                <Tooltip direction="top" offset={[0, -5]}>
                  <span>{ev.event_id.replace('EV_SYNTH_', 'EV-')}: {ev.max_intensity.toFixed(1)} mm</span>
                </Tooltip>
              </CircleMarker>
            )
          })}

          <ZoomButtons />
        </MapContainer>

        {/* Floating Left Vertical Tool Controls Bar */}
        <div
          className="map-vertical-toolbar"
          role="toolbar"
          aria-label="Map Tools"
          onMouseDown={stopToolbarPropagation}
          onDoubleClick={stopToolbarPropagation}
          onTouchStart={stopToolbarPropagation}
        >
          <button
            type="button"
            className={`toolbar-btn ${showLayers ? 'active' : ''}`}
            onClick={() => setShowLayers((prev) => !prev)}
            title="Toggle Map Display Layers"
            aria-expanded={showLayers}
          >
            <LayersIcon size={14} />
            <span>Layers</span>
          </button>

          <button
            type="button"
            className={`toolbar-btn ${showTrack ? 'active' : ''}`}
            onClick={() => setShowTrack((prev) => !prev)}
            title="Toggle Trajectory Track"
            aria-pressed={showTrack}
          >
            <RouteIcon size={14} />
            <span>Track</span>
          </button>

          <button
            type="button"
            className={`toolbar-btn risk-active ${showRisk ? 'active' : ''}`}
            onClick={() => setShowRisk((prev) => !prev)}
            title="Toggle High-Risk Cyclone Envelopes"
            aria-pressed={showRisk}
          >
            <FlameIcon size={14} />
            <span>Risk</span>
          </button>

          <button
            type="button"
            className={`toolbar-btn ${showUncertainty ? 'active' : ''}`}
            onClick={() => setShowUncertainty((prev) => !prev)}
            title="Toggle Uncertainty Spread Cone"
            aria-pressed={showUncertainty}
          >
            <CrosshairIcon size={14} />
            <span>Uncertainty</span>
          </button>

          <button
            type="button"
            className={`toolbar-btn ${tileMode === 'satellite' ? 'active' : ''}`}
            onClick={() => setTileMode((prev) => (prev === 'satellite' ? 'dark' : 'satellite'))}
            title="Toggle Satellite vs Dark Base Map"
          >
            <SatelliteIcon size={14} />
            <span>Satellite</span>
          </button>
        </div>

        {/* Interactive Layers Menu Dropdown */}
        {showLayers && (
          <div
            className="map-layers-menu"
            role="dialog"
            aria-label="Layer Settings"
            onMouseDown={stopToolbarPropagation}
            onClick={stopToolbarPropagation}
          >
            <div className="layers-menu-header">
              <span className="layers-menu-title">Display Layers</span>
              <button
                type="button"
                className="layers-close-btn"
                onClick={() => setShowLayers(false)}
                aria-label="Close layers menu"
              >
                &times;
              </button>
            </div>
            <div className="layers-menu-options">
              <label className="layers-menu-item">
                <span>Satellite Imagery</span>
                <input
                  type="checkbox"
                  checked={tileMode === 'satellite'}
                  onChange={() => setTileMode((prev) => (prev === 'satellite' ? 'dark' : 'satellite'))}
                />
              </label>
              <label className="layers-menu-item">
                <span>Cyclone Radar Swirl</span>
                <input
                  type="checkbox"
                  checked={showRisk}
                  onChange={() => setShowRisk((prev) => !prev)}
                />
              </label>
              <label className="layers-menu-item">
                <span>Storm Trajectory Track</span>
                <input
                  type="checkbox"
                  checked={showTrack}
                  onChange={() => setShowTrack((prev) => !prev)}
                />
              </label>
              <label className="layers-menu-item">
                <span>Uncertainty Cone</span>
                <input
                  type="checkbox"
                  checked={showUncertainty}
                  onChange={() => setShowUncertainty((prev) => !prev)}
                />
              </label>
              <label className="layers-menu-item">
                <span>Political Boundaries</span>
                <input
                  type="checkbox"
                  checked={showBorders}
                  onChange={() => setShowBorders((prev) => !prev)}
                />
              </label>
            </div>
          </div>
        )}

        {/* In-Map Bottom-Right Rainfall Legend */}
        <div className="map-rainfall-legend">
          <div className="rainfall-legend-title">Rainfall (mm)</div>
          <div className="rainfall-legend-list">
            <div className="rainfall-legend-row">
              <span className="legend-swatch" style={{ backgroundColor: '#990000' }} />
              <span>&gt; 300</span>
            </div>
            <div className="rainfall-legend-row">
              <span className="legend-swatch" style={{ backgroundColor: '#e11d48' }} />
              <span>200 &ndash; 300</span>
            </div>
            <div className="rainfall-legend-row">
              <span className="legend-swatch" style={{ backgroundColor: '#f59e0b' }} />
              <span>100 &ndash; 200</span>
            </div>
            <div className="rainfall-legend-row">
              <span className="legend-swatch" style={{ backgroundColor: '#22c55e' }} />
              <span>50 &ndash; 100</span>
            </div>
            <div className="rainfall-legend-row">
              <span className="legend-swatch" style={{ backgroundColor: '#06b6d4' }} />
              <span>20 &ndash; 50</span>
            </div>
            <div className="rainfall-legend-row">
              <span className="legend-swatch" style={{ backgroundColor: '#2563eb' }} />
              <span>10 &ndash; 20</span>
            </div>
            <div className="rainfall-legend-row">
              <span className="legend-swatch" style={{ backgroundColor: '#1e3a8a' }} />
              <span>&lt; 10</span>
            </div>
          </div>
        </div>

        {/* In-Map Bottom-Left Scale Bar */}
        <div className="map-bottom-scale-bar">
          <div className="scale-marks">
            <span>0</span>
            <span>250</span>
            <span>500 km</span>
          </div>
          <div className="scale-line-graphic" />
        </div>
      </div>
    </div>
  )
}

