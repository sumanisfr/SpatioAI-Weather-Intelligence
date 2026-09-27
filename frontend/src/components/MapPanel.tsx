import { useEffect } from 'react'
import { MapContainer, TileLayer, CircleMarker, Rectangle, Polyline, GeoJSON, useMap } from 'react-leaflet'
import type { LatLngBoundsExpression } from 'leaflet'
import type { Event, RiskResponse, Track } from '../types/api'
import 'leaflet/dist/leaflet.css'

type Props = { event: Event | null; track: Track | null; risk: RiskResponse | null; showTrack: boolean; showRisk: boolean }

function Viewport({ event }: { event: Event | null }) {
  const map = useMap()
  useEffect(() => {
    if (event) map.flyTo([event.centroid_lat, event.centroid_lon], 6, { duration: 0.6 })
  }, [event, map])
  return null
}

export function MapPanel({ event, track, risk, showTrack, showRisk }: Props) {
  const center: [number, number] = event ? [event.centroid_lat, event.centroid_lon] : [20.5, 78.9]
  const bounds: LatLngBoundsExpression | undefined = event ? [[event.bbox[0], event.bbox[2]], [event.bbox[1], event.bbox[3]]] : undefined
  const trajectory = track?.centroids.map(([lat, lon]) => [lat, lon] as [number, number]) ?? []
  return (
    <div className="map-shell">
      <MapContainer center={center} zoom={event ? 6 : 5} scrollWheelZoom className="map-canvas">
        <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        <Viewport event={event} />
        {event && <>
          {bounds && <Rectangle bounds={bounds} pathOptions={{ color: '#f59e0b', weight: 2, fillColor: '#f59e0b', fillOpacity: 0.08 }} />}
          <CircleMarker center={[event.centroid_lat, event.centroid_lon]} radius={8} pathOptions={{ color: '#0f766e', fillColor: '#14b8a6', fillOpacity: 0.95 }} />
        </>}
        {showTrack && trajectory.length > 1 && <Polyline positions={trajectory} pathOptions={{ color: '#2563eb', weight: 3, dashArray: '8 6' }} />}
        {showRisk && risk?.geojson && <GeoJSON data={risk.geojson} style={{ color: '#ef4444', weight: 1, fillColor: '#fb7185', fillOpacity: 0.25 }} />}
      </MapContainer>
      <div className="map-legend">
        <span><i className="legend-dot event-dot" /> selected event</span>
        <span><i className="legend-line track-line" /> track trajectory</span>
        <span><i className="legend-box risk-box" /> empirical risk footprint</span>
      </div>
    </div>
  )
}
