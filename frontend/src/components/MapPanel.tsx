import { useEffect, useState, useMemo, useRef } from 'react'
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Rectangle,
  Polyline,
  GeoJSON,
  Popup,
  Tooltip,
  useMap,
  useMapEvents,
} from 'react-leaflet'
import type { LatLngBoundsExpression } from 'leaflet'
import {
  LayersIcon,
  CrosshairIcon,
  TargetIcon,
  MaximizeIcon,
  MinimizeIcon,
  EyeIcon,
} from './Icons'
import type { Event, RiskResponse, Track } from '../types/api'
import 'leaflet/dist/leaflet.css'

type MapPanelProps = {
  allEvents: Event[]
  event: Event | null
  track: Track | null
  risk: RiskResponse | null
  onSelectEvent: (eventId: string) => void
  activeBasin?: string
}

type TileMode = 'dark' | 'satellite' | 'street'

// Meteorological Basin Focus Coordinates (Calibrated for Indian Subcontinent & Bay of Bengal)
const BASIN_COORDS: Record<string, { center: [number, number]; zoom: number }> = {
  subcontinent: { center: [19.5, 82.5], zoom: 5 },
  bay_of_bengal: { center: [17.0, 88.5], zoom: 6 },
  peninsular: { center: [14.0, 77.5], zoom: 6 },
  western_ghats: { center: [18.5, 73.5], zoom: 6 },
}

function ViewportManager({
  event,
  activeBasin,
  targetOverride,
}: {
  event: Event | null
  activeBasin?: string
  targetOverride: { center: [number, number]; zoom: number; key: number } | null
}) {
  const map = useMap()
  const prevEventIdRef = useRef<string | null>(null)
  const prevBasinRef = useRef<string | undefined>(activeBasin)
  const isFirstMountRef = useRef<boolean>(true)

  useEffect(() => {
    // Invalidate map size on initial mount and resize to ensure correct pixel dimensions
    const timer = setTimeout(() => {
      map.invalidateSize()
      if (isFirstMountRef.current) {
        isFirstMountRef.current = false
        if (event) {
          prevEventIdRef.current = event.event_id
          map.setView([event.centroid_lat, event.centroid_lon], 6)
        } else if (activeBasin && BASIN_COORDS[activeBasin]) {
          const b = BASIN_COORDS[activeBasin]
          map.setView(b.center, b.zoom)
        }
      }
    }, 150)
    return () => clearTimeout(timer)
  }, [map, event, activeBasin])

  useEffect(() => {
    if (targetOverride) {
      map.flyTo(targetOverride.center, targetOverride.zoom, { duration: 0.8, easeLinearity: 0.25 })
      return
    }

    const basinChanged = activeBasin !== undefined && activeBasin !== prevBasinRef.current
    if (basinChanged) {
      prevBasinRef.current = activeBasin
      if (activeBasin && BASIN_COORDS[activeBasin]) {
        const b = BASIN_COORDS[activeBasin]
        map.flyTo(b.center, b.zoom, { duration: 0.8, easeLinearity: 0.25 })
        return
      }
    }

    const eventChanged = event && event.event_id !== prevEventIdRef.current
    if (eventChanged && event) {
      prevEventIdRef.current = event.event_id
      map.flyTo([event.centroid_lat, event.centroid_lon], 6, { duration: 0.8, easeLinearity: 0.25 })
    }
  }, [event, activeBasin, targetOverride, map])

  return null
}

function CursorTracker({
  onMove,
  onZoomChange,
}: {
  onMove: (lat: number, lon: number) => void
  onZoomChange: (z: number) => void
}) {
  useMapEvents({
    mousemove(e) {
      onMove(e.latlng.lat, e.latlng.lng)
    },
    zoomend(e) {
      onZoomChange(e.target.getZoom())
    },
  })
  return null
}

const OPENWEATHER_KEY = (import.meta.env.VITE_OPENWEATHER_API_KEY as string | undefined) || 'babf0f4e684e15176c2fc62d0a394fac'

export function MapPanel({
  allEvents,
  event,
  track,
  risk,
  onSelectEvent,
  activeBasin = 'subcontinent',
}: MapPanelProps) {
  // Layer visibility controls
  const [showTrack, setShowTrack] = useState(true)
  const [showFootprint, setShowFootprint] = useState(true)
  const [showRisk, setShowRisk] = useState(true)
  const [showOtherEvents, setShowOtherEvents] = useState(true)
  const [showLiveRadar, setShowLiveRadar] = useState(false)
  const [showLayersMenu, setShowLayersMenu] = useState(false)
  const [showLegend, setShowLegend] = useState(true)
  const [isExpanded, setIsExpanded] = useState(false)
  const [tileMode, setTileMode] = useState<TileMode>('dark')

  // Cursor & Zoom HUD state
  const [cursorCoords, setCursorCoords] = useState<[number, number] | null>(null)
  const [currentZoom, setCurrentZoom] = useState<number>(5)

  // Target override
  const [targetOverride, setTargetOverride] = useState<{ center: [number, number]; zoom: number; key: number } | null>(null)
  const [overrideCounter, setOverrideCounter] = useState(0)

  // Optimal center capturing Indian Subcontinent & Bay of Bengal Basin
  const defaultCenter: [number, number] = [19.0, 83.5]
  const center: [number, number] = event ? [event.centroid_lat, event.centroid_lon] : defaultCenter

  // Bounding box for selected event
  const bounds: LatLngBoundsExpression | undefined = useMemo(() => {
    if (!event) return undefined
    return [
      [event.bbox[0], event.bbox[2]],
      [event.bbox[1], event.bbox[3]],
    ]
  }, [event])

  // Storm trajectory track
  const trajectory = useMemo(() => {
    return track?.centroids.map(([lat, lon]) => [lat, lon] as [number, number]) ?? []
  }, [track])

  // Free, high-performance basemap configurations with ZERO watermarks
  const currentTiles = useMemo(() => {
    switch (tileMode) {
      case 'satellite':
        return {
          base: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
          labels: null,
          attr: '&copy; Esri &mdash; Earthstar Geographics',
        }
      case 'street':
        return {
          base: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
          labels: null,
          attr: '&copy; OpenStreetMap contributors',
        }
      case 'dark':
      default:
        return {
          base: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
          labels: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
          attr: '&copy; Esri, DeLorme, NAVTEQ',
        }
    }
  }, [tileMode])

  const handleResetToEvent = () => {
    if (event) {
      const next = overrideCounter + 1
      setOverrideCounter(next)
      setTargetOverride({ center: [event.centroid_lat, event.centroid_lon], zoom: 6.5, key: next })
    }
  }

  return (
    <div className={`map-card-wrapper ${isExpanded ? 'fullscreen-map' : ''}`}>
      {/* Precision Map Canvas */}
      <div className="map-canvas-container">
        <MapContainer
          center={center}
          zoom={event ? 6 : 5}
          minZoom={4}
          maxZoom={16}
          maxBounds={[
            [-5.0, 50.0],
            [40.0, 115.0],
          ]}
          maxBoundsViscosity={0.75}
          scrollWheelZoom={true}
          className="leaflet-dark-container"
        >
          {/* Base Tiles (Esri Dark Canvas by default) */}
          <TileLayer attribution={currentTiles.attr} url={currentTiles.base} maxZoom={16} minZoom={4} />

          {/* Reference Labels (Crisp geographical names) */}
          {currentTiles.labels && (
            <TileLayer attribution="" url={currentTiles.labels} maxZoom={16} />
          )}

          {/* Optional Live Precipitation Radar from OpenWeatherMap */}
          {showLiveRadar && OPENWEATHER_KEY && (
            <TileLayer
              attribution="&copy; OpenWeatherMap Rain Radar"
              url={`https://tile.openweathermap.org/map/precipitation_new/{z}/{x}/{y}.png?appid=${OPENWEATHER_KEY}`}
              opacity={0.65}
              maxZoom={16}
            />
          )}

          <ViewportManager
            event={event}
            activeBasin={activeBasin}
            targetOverride={targetOverride}
          />

          <CursorTracker
            onMove={(lat, lon) => setCursorCoords([lat, lon])}
            onZoomChange={(z) => setCurrentZoom(z)}
          />

          {/* Other events (inactive clusters) */}
          {showOtherEvents &&
            allEvents.map((ev) => {
              if (event && ev.event_id === event.event_id) return null
              const isExtreme = ev.max_intensity >= 110
              const isHeavy = ev.max_intensity >= 80 && ev.max_intensity < 110

              // Severity-distinct colors: extreme=red, severe=amber, moderate=cyan
              const color = isExtreme ? '#ef4444' : isHeavy ? '#f59e0b' : '#38bdf8'
              const fillColor = isExtreme ? '#dc2626' : isHeavy ? '#d97706' : '#0284c7'

              return (
                <CircleMarker
                  key={ev.event_id}
                  center={[ev.centroid_lat, ev.centroid_lon]}
                  radius={6}
                  pathOptions={{
                    color,
                    fillColor,
                    fillOpacity: 0.8,
                    weight: 1.5,
                  }}
                  eventHandlers={{
                    click: () => onSelectEvent(ev.event_id),
                  }}
                >
                  <Tooltip direction="top" offset={[0, -6]} opacity={0.96}>
                    <div className="map-tooltip-content">
                      <div className="tooltip-event-code">{ev.event_id.replace('EV_SYNTH_', 'EV-')}</div>
                      <div className="tooltip-rate text-amber-400">{ev.max_intensity.toFixed(1)} mm/h</div>
                      <div className="tooltip-area">{ev.area_km2.toLocaleString()} km&sup2;</div>
                    </div>
                  </Tooltip>
                </CircleMarker>
              )
            })}

          {/* Storm trajectory path */}
          {showTrack && trajectory.length > 1 && (
            <>
              <Polyline
                positions={trajectory}
                pathOptions={{
                  color: '#a78bfa',
                  weight: 2.5,
                  dashArray: '8 5',
                  opacity: 0.85,
                }}
              />
              {/* Waypoint markers along the trajectory */}
              {trajectory.map((pos, idx) => (
                <CircleMarker
                  key={`waypoint-${idx}`}
                  center={pos}
                  radius={4}
                  pathOptions={{
                    color: '#c4b5fd',
                    fillColor: '#7c3aed',
                    fillOpacity: 1,
                    weight: 1.5,
                  }}
                >
                  <Tooltip direction="right" offset={[6, 0]} opacity={0.95}>
                    <div className="map-tooltip-content">
                      <strong>T+{idx * 6}h Waypoint</strong>
                      <div>{pos[0].toFixed(2)}&deg;N, {pos[1].toFixed(2)}&deg;E</div>
                    </div>
                  </Tooltip>
                </CircleMarker>
              ))}
            </>
          )}

          {/* Spatial footprint bounding box */}
          {showFootprint && bounds && (
            <Rectangle
              bounds={bounds}
              pathOptions={{
                color: '#f59e0b',
                weight: 1.5,
                fillColor: '#f59e0b',
                fillOpacity: 0.06,
                dashArray: '4 4',
              }}
            />
          )}

          {/* Empirical risk GeoJSON footprint */}
          {showRisk && risk?.geojson && (
            <GeoJSON
              key={`risk-${risk.event_id}-${risk.threshold}`}
              data={risk.geojson}
              style={{
                color: '#ef4444',
                weight: 1.5,
                fillColor: '#f43f5e',
                fillOpacity: 0.25,
              }}
            />
          )}

          {/* Selected event beacon with pulse marker */}
          {event && (
            <CircleMarker
              center={[event.centroid_lat, event.centroid_lon]}
              radius={11}
              pathOptions={{
                color: '#22d3ee',
                fillColor: '#06b6d4',
                fillOpacity: 0.95,
                weight: 3,
              }}
            >
              <Popup>
                <div className="leaflet-custom-popup">
                  <div className="popup-header">
                    <span className="popup-tag">CENTROID TELEMETRY</span>
                    <h3>{event.event_id.replace('EV_SYNTH_', 'EV-')}</h3>
                  </div>
                  <div className="popup-body">
                    <div className="popup-row"><span>Track:</span><strong>{event.track_id ? event.track_id.replace('TRK_SYNTH_', 'TRK-') : 'Untracked'}</strong></div>
                    <div className="popup-row"><span>Position:</span><strong>{event.centroid_lat.toFixed(2)}&deg;N, {event.centroid_lon.toFixed(2)}&deg;E</strong></div>
                    <div className="popup-row"><span>Peak Rain:</span><strong className="text-amber-400">{event.max_intensity.toFixed(1)} mm/h</strong></div>
                    <div className="popup-row"><span>Mean Rain:</span><strong>{event.mean_intensity.toFixed(1)} mm/h</strong></div>
                    <div className="popup-row"><span>Footprint:</span><strong>{event.area_km2.toLocaleString()} km&sup2;</strong></div>
                    <div className="popup-row"><span>Timestamp:</span><strong>{new Date(event.timestamp).toUTCString().slice(5, 22)}</strong></div>
                  </div>
                </div>
              </Popup>
            </CircleMarker>
          )}
        </MapContainer>

        {/* Floating Top Controls Bar (Top-Right) */}
        <div className="map-floating-controls-top" role="toolbar" aria-label="Map Controls">
          {/* Basemap Switcher */}
          <div className="map-segmented-btn-group">
            <button
              className={`map-seg-btn ${tileMode === 'dark' ? 'active' : ''}`}
              onClick={() => setTileMode('dark')}
              title="Cartographic Dark Gray Basemap"
              type="button"
            >
              Dark
            </button>
            <button
              className={`map-seg-btn ${tileMode === 'satellite' ? 'active' : ''}`}
              onClick={() => setTileMode('satellite')}
              title="Esri World Imagery (Satellite)"
              type="button"
            >
              Satellite
            </button>
            <button
              className={`map-seg-btn ${tileMode === 'street' ? 'active' : ''}`}
              onClick={() => setTileMode('street')}
              title="OpenStreetMap Topography"
              type="button"
            >
              Street
            </button>
          </div>

          {/* Layers Popover Toggle */}
          <div className="relative">
            <button
              className={`map-icon-toggle-btn ${showLayersMenu ? 'active' : ''}`}
              onClick={() => setShowLayersMenu(!showLayersMenu)}
              title="Toggle Overlays & Layers"
              type="button"
            >
              <LayersIcon size={14} />
              <span>Layers</span>
            </button>

            {showLayersMenu && (
              <div className="map-layers-popover" role="dialog" aria-label="Layer Settings">
                <div className="layers-popover-header">MAP OVERLAYS</div>
                <label className="layer-checkbox-row">
                  <input
                    type="checkbox"
                    checked={showTrack}
                    onChange={(e) => setShowTrack(e.target.checked)}
                  />
                  <span>Storm Trajectory Track</span>
                </label>
                <label className="layer-checkbox-row">
                  <input
                    type="checkbox"
                    checked={showFootprint}
                    onChange={(e) => setShowFootprint(e.target.checked)}
                  />
                  <span>Physical Bounding Box</span>
                </label>
                <label className="layer-checkbox-row">
                  <input
                    type="checkbox"
                    checked={showRisk}
                    onChange={(e) => setShowRisk(e.target.checked)}
                  />
                  <span>Exceedance Risk Polygon</span>
                </label>
                <label className="layer-checkbox-row">
                  <input
                    type="checkbox"
                    checked={showOtherEvents}
                    onChange={(e) => setShowOtherEvents(e.target.checked)}
                  />
                  <span>Other Candidate Clusters</span>
                </label>
                <label className="layer-checkbox-row highlight">
                  <input
                    type="checkbox"
                    checked={showLiveRadar}
                    onChange={(e) => setShowLiveRadar(e.target.checked)}
                  />
                  <span className="text-emerald-400">Live Rain Radar</span>
                </label>
              </div>
            )}
          </div>

          {/* Selected Event Focus Button */}
          {event && (
            <button
              className="map-icon-toggle-btn"
              onClick={handleResetToEvent}
              title={`Focus on ${event.event_id.replace('EV_SYNTH_', 'EV-')}`}
              type="button"
            >
              <TargetIcon size={13} className="text-amber-400" />
              <span>{event.event_id.replace('EV_SYNTH_', 'EV-')}</span>
            </button>
          )}

          {/* Fullscreen Expand Toggle */}
          <button
            className="map-icon-toggle-btn icon-only"
            onClick={() => setIsExpanded(!isExpanded)}
            title={isExpanded ? 'Restore Map View' : 'Maximize Map Canvas'}
            aria-label="Toggle Fullscreen Map View"
            type="button"
          >
            {isExpanded ? <MinimizeIcon size={14} /> : <MaximizeIcon size={14} />}
          </button>
        </div>

        {/* Floating Top-Left Domain HUD */}
        <div className="map-domain-hud">
          <CrosshairIcon size={12} className="text-sky-400" />
          <span>Bay of Bengal &amp; Subcontinent Radar Grid</span>
        </div>

        {/* Collapsible Map Legend (Bottom-Right) */}
        <div className="map-floating-legend">
          <div
            className="legend-strip-header"
            onClick={() => setShowLegend(!showLegend)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                setShowLegend(!showLegend)
              }
            }}
            role="button"
            tabIndex={0}
            title={showLegend ? 'Click to collapse GIS legend' : 'Click to expand GIS legend'}
          >
            <span className="legend-strip-title">LEGEND</span>
            <span className="legend-strip-toggle">{showLegend ? 'Hide' : 'Show'}</span>
          </div>

          {showLegend && (
            <div className="legend-strip-body">
              <div className="legend-entry">
                <span className="legend-dot cyan-halo" />
                <span>Selected Anomaly</span>
              </div>
              <div className="legend-entry">
                <span className="legend-dot red" />
                <span>&gt;110 mm/h Extreme</span>
              </div>
              <div className="legend-entry">
                <span className="legend-dot amber" />
                <span>80&ndash;110 mm/h Heavy</span>
              </div>
              <div className="legend-entry">
                <span className="legend-dot blue" />
                <span>&lt;80 mm/h Moderate</span>
              </div>
              <div className="legend-entry">
                <span className="legend-dash indigo" />
                <span>Storm Track (6h)</span>
              </div>
              <div className="legend-entry">
                <span className="legend-box amber-dash" />
                <span>Bounding Box</span>
              </div>
              <div className="legend-entry">
                <span className="legend-box red-fill" />
                <span>Risk Zone</span>
              </div>
              {showLiveRadar && (
                <div className="legend-entry">
                  <span className="legend-dot emerald" />
                  <span>Rain Radar Tile</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Precision Coordinates HUD (Bottom-Left) */}
        <div className="map-coordinates-hud">
          {cursorCoords ? (
            <span className="hud-coord-item">
              <EyeIcon size={11} className="text-sky-400 inline mr-1" />
              {cursorCoords[0].toFixed(3)}&deg;N, {cursorCoords[1].toFixed(3)}&deg;E
            </span>
          ) : (
            <span className="hud-coord-item text-slate-500">
              Hover cursor for coordinate telemetry
            </span>
          )}
          <span className="hud-divider">&bull;</span>
          <span className="hud-zoom">Zoom: {currentZoom.toFixed(1)}x</span>
          <span className="hud-divider">&bull;</span>
          <span className="hud-crs">WGS84</span>
        </div>
      </div>
    </div>
  )
}
