import { useEffect, useState } from 'react'
import { SatelliteIcon, WindIcon, RefreshCwIcon, MapPinIcon } from './Icons'
import { fetchLiveStationWeather, type LiveStationWeather } from '../api/liveWeather'
import { haversineDistanceKm } from '../utils/geo'
import type { Event } from '../types/api'

type LiveStationCardProps = {
  event: Event | null
}

export function LiveStationCard({ event }: LiveStationCardProps) {
  const [weather, setWeather] = useState<LiveStationWeather | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!event) return
    let active = true
    const controller = new AbortController()

    const fetchStation = async () => {
      try {
        const data = await fetchLiveStationWeather(event.centroid_lat, event.centroid_lon, controller.signal)
        if (active) {
          setWeather(data)
          setLoading(false)
        }
      } catch (err) {
        if (active && !controller.signal.aborted) {
          setError(err instanceof Error ? err.message : 'Live meteorological station data unavailable')
          setLoading(false)
        }
      }
    }

    queueMicrotask(() => {
      if (active) {
        setLoading(true)
        setError(null)
        void fetchStation()
      }
    })

    return () => {
      active = false
      controller.abort()
    }
  }, [event])

  const handleManualRefresh = () => {
    if (!event) return
    setLoading(true)
    setError(null)
    void fetchLiveStationWeather(event.centroid_lat, event.centroid_lon)
      .then((data) => {
        setWeather(data)
        setLoading(false)
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : 'Station data refresh failed')
        setLoading(false)
      })
  }

  if (!event) return null

  // Geodesic distance to reporting physical surface station
  const distanceKm = weather
    ? Math.max(1, Math.round(haversineDistanceKm(event.centroid_lat, event.centroid_lon, weather.stationLat, weather.stationLon)))
    : null

  return (
    <section className="card-panel station-card" aria-label="Physical Observation Ground-Truth">
      <div className="card-header-row">
        <div>
          <span className="panel-eyebrow">PHYSICAL OBSERVATION VALIDATION</span>
          <h3 className="panel-title">Surface Station Ground-Truth</h3>
        </div>
        <div className="flex items-center gap-2">
          <div className="station-source-tag">
            <SatelliteIcon size={12} className="text-emerald-400" />
            <span>{weather?.source ? `${weather.source} Real Observation` : 'Physical Station Ground-Truth'}</span>
          </div>
          <button
            className="action-icon-btn-subtle"
            onClick={handleManualRefresh}
            disabled={loading}
            title="Poll fresh meteorological observation from station"
            type="button"
          >
            <RefreshCwIcon size={12} className={loading ? 'animate-spin text-sky-400' : 'text-slate-400'} />
          </button>
        </div>
      </div>

      {loading && (
        <div className="state-placeholder">
          <span className="loading-spinner" />
          <span className="text-xs text-slate-400">
            Polling surface station near {event.centroid_lat.toFixed(2)}&deg;N, {event.centroid_lon.toFixed(2)}&deg;E...
          </span>
        </div>
      )}

      {error && (
        <div className="state-placeholder error">
          <span className="text-xs text-rose-300">{error}</span>
        </div>
      )}

      {!loading && !error && weather && (
        <div className="station-content-body">
          {/* Station Hero Bar */}
          <div className="station-hero-strip">
            <div className="station-location-info">
              <span className="station-name-title">
                <MapPinIcon size={13} className="inline mr-1 text-emerald-400" />
                {weather.stationName}, {weather.country}
              </span>
              <span className="station-condition-subtitle">{weather.description.toUpperCase()}</span>
            </div>
            <div className="station-temperature-block">
              <span className="station-temp-degrees">{Math.round(weather.temp)}&deg;C</span>
              <span className="station-feels-degrees">Feels {Math.round(weather.feelsLike)}&deg;C</span>
            </div>
          </div>

          {/* 4-Column Sensor Readout Grid */}
          <div className="station-readouts-grid">
            <div className="readout-cell">
              <span className="readout-label">Relative Humidity</span>
              <span className="readout-val text-sky-400">{weather.humidity}%</span>
            </div>
            <div className="readout-cell">
              <span className="readout-label">Atm. Pressure</span>
              <span className="readout-val">{weather.pressure} <small className="text-slate-500">hPa</small></span>
            </div>
            <div className="readout-cell">
              <span className="readout-label">Surface Wind</span>
              <span className="readout-val">
                <WindIcon size={11} className="inline mr-1 text-teal-400" />
                {weather.windSpeed.toFixed(1)} <small className="text-slate-500">m/s ({weather.windDeg}&deg;)</small>
              </span>
            </div>
            <div className="readout-cell">
              <span className="readout-label">Cloud Cover</span>
              <span className="readout-val text-amber-400">{weather.clouds}%</span>
            </div>
          </div>

          {/* Alignment Note */}
          <div className="station-footnote">
            <span>
              Real physical station reporting from <strong>{weather.stationName}</strong> (~{distanceKm ?? '12'} km from centroid {event.centroid_lat.toFixed(2)}&deg;N, {event.centroid_lon.toFixed(2)}&deg;E). Validates NWP simulated fields against actual ground-truth sensors.
            </span>
          </div>
        </div>
      )}
    </section>
  )
}
