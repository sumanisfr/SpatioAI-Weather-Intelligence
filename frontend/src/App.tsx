import { useMemo, useState } from 'react'
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { MapPanel } from './components/MapPanel'
import { useDashboardData } from './hooks/useDashboardData'
import './App.css'

function formatTime(value: string) {
  return new Intl.DateTimeFormat(undefined, { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}

function metric(value: number, digits = 1) { return Number.isFinite(value) ? value.toFixed(digits) : '—' }

function App() {
  const data = useDashboardData()
  const [query, setQuery] = useState('')
  const [showTrack, setShowTrack] = useState(true)
  const [showRisk, setShowRisk] = useState(true)
  const filteredEvents = useMemo(() => data.events.filter((event) => `${event.event_id} ${event.track_id ?? ''}`.toLowerCase().includes(query.toLowerCase())), [data.events, query])
  const trackTimeline = data.selectedTrack?.event_ids.map((eventId, index) => ({
    step: index + 1,
    intensity: data.events.find((item) => item.event_id === eventId)?.max_intensity ?? 0,
  })) ?? []

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">◌</div>
          <div><div className="brand-name">SpatioAI</div><div className="brand-subtitle">AI-DRIVEN WEATHER INTELLIGENCE</div></div>
        </div>
        <div className="header-actions">
          <span className="demo-badge"><span className="status-dot" /> RESEARCH / SYNTHETIC VALIDATION</span>
          <a className="docs-link" href="http://localhost:8000/docs" target="_blank" rel="noreferrer">API docs ↗</a>
          <button className="icon-button" onClick={() => void data.refresh()} title="Refresh backend data" aria-label="Refresh backend data">↻</button>
        </div>
      </header>

      <section className="status-strip">
        <div className="status-item"><span className={`status-pill ${data.health ? 'online' : 'offline'}`}><span className="status-dot" /> {data.health ? 'Backend online' : 'Backend offline'}</span><span className="muted">API v{data.health?.version ?? '—'}</span></div>
        <div className="status-item"><span aria-hidden="true">✓</span><span>{data.health?.models_loaded ? 'Model registry ready' : 'Model readiness pending'}</span></div>
        <div className="status-item"><span aria-hidden="true">◈</span><span>{data.events.length} detected events</span></div>
        <div className="status-item"><span aria-hidden="true">↝</span><span>{data.tracks.length} tracked systems</span></div>
      </section>

      <div className="dashboard-grid">
        <aside className="sidebar panel">
          <div className="panel-heading"><div><div className="eyebrow">EVENT MONITOR</div><h1>Detected events</h1></div><span className="count-chip">{filteredEvents.length}</span></div>
          <div className="search-wrap"><span aria-hidden="true">⌕</span><input aria-label="Filter events" placeholder="Filter event or track" value={query} onChange={(event) => setQuery(event.target.value)} /></div>
          {data.loading && <div className="state-block">Loading events...</div>}
          {data.error && <div className="state-block error-state"><span aria-hidden="true">!</span><span>{data.error}</span></div>}
          {!data.loading && !data.error && filteredEvents.length === 0 && <div className="state-block">No events match this filter.</div>}
          <div className="event-list">
            {filteredEvents.map((event) => <button className={`event-row ${data.selectedId === event.event_id ? 'selected' : ''}`} key={event.event_id} onClick={() => data.setSelectedId(event.event_id)}>
              <div className="event-row-top"><span className="event-id">{event.event_id.replace('EV_SYNTH_', 'EV ')}</span><span className="event-time">{formatTime(event.timestamp)}</span></div>
              <div className="event-row-meta"><span>⌖ {event.centroid_lat.toFixed(2)}°, {event.centroid_lon.toFixed(2)}°</span><span>{event.max_intensity.toFixed(0)} mm</span></div>
              <div className="event-row-track">{event.track_id ?? 'Track unavailable'}</div>
            </button>)}
          </div>
          <div className="sidebar-footer">Source mode: <strong>{data.events.length ? 'synthetic_demo' : '—'}</strong></div>
        </aside>

        <section className="main-column">
          <div className="map-panel panel"><div className="map-header"><div><div className="eyebrow">SPATIAL CONTEXT</div><h2>Event footprint & trajectory</h2></div><div className="layer-controls"><label><input type="checkbox" checked={showTrack} onChange={(event) => setShowTrack(event.target.checked)} /> Track</label><label><input type="checkbox" checked={showRisk} onChange={(event) => setShowRisk(event.target.checked)} /> Risk</label></div></div><MapPanel event={data.selectedEvent} track={data.selectedTrack} risk={data.risk} showTrack={showTrack} showRisk={showRisk} /></div>

          <div className="metrics-grid">
            <section className="panel detail-panel"><div className="eyebrow">SELECTED EVENT</div><h2>{data.selectedEvent?.event_id.replace('EV_SYNTH_', 'EV ') ?? 'No event selected'}</h2>{data.selectedEvent ? <div className="detail-grid"><div><span>Timestamp</span><strong>{formatTime(data.selectedEvent.timestamp)}</strong></div><div><span>Track</span><strong>{data.selectedEvent.track_id ?? 'Unavailable'}</strong></div><div><span>Centroid</span><strong>{data.selectedEvent.centroid_lat.toFixed(2)}°, {data.selectedEvent.centroid_lon.toFixed(2)}°</strong></div><div><span>Footprint</span><strong>{data.selectedEvent.area_km2.toFixed(0)} km²</strong></div><div><span>Maximum intensity</span><strong>{data.selectedEvent.max_intensity.toFixed(1)} mm</strong></div><div><span>Mean intensity</span><strong>{data.selectedEvent.mean_intensity.toFixed(1)} mm</strong></div></div> : <div className="muted">Select an event from the monitor.</div>}</section>
            <section className="panel model-panel"><div className="eyebrow">MODEL STATUS</div><h2>Inference registry</h2>{data.health ? <div className="model-list">{Object.entries(data.health.models).map(([name, status]) => <div className="model-row" key={name}><span className={`model-dot ${status.loaded ? 'ready' : 'unavailable'}`} /><span className="model-name">{name.replace('_', ' ')}</span><span className="model-state">{status.loaded ? 'ready' : 'unavailable'}</span></div>)}</div> : <div className="muted">Waiting for backend...</div>}</section>
          </div>

          <div className="lower-grid">
            <section className="panel timeline-panel"><div className="panel-heading"><div><div className="eyebrow">TRACK SIGNAL</div><h2>Intensity timeline</h2></div>{data.selectedTrack && <span className="track-chip">{data.selectedTrack.track_id}</span>}</div>{trackTimeline.length ? <ResponsiveContainer width="100%" height={180}><LineChart data={trackTimeline} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}><XAxis dataKey="step" stroke="#728096" tickLine={false} axisLine={false} /><YAxis stroke="#728096" tickLine={false} axisLine={false} unit=" mm" width={52} /><Tooltip contentStyle={{ background: '#101a2a', border: '1px solid #2b3b52', borderRadius: 8, color: '#eef4fa' }} /><Line type="monotone" dataKey="intensity" stroke="#f59e0b" strokeWidth={3} dot={{ r: 4, fill: '#f59e0b' }} /></LineChart></ResponsiveContainer> : <div className="state-block">Track trajectory unavailable.</div>}</section>
            <section className="panel risk-panel"><div className="eyebrow">EXPERIMENTAL RISK ESTIMATE</div><h2>Threshold exceedance</h2>{data.riskLoading && <div className="state-block">Analyzing risk...</div>}{data.riskError && <div className="state-block error-state"><span aria-hidden="true">!</span>{data.riskError}</div>}{data.risk && <div className="risk-content"><div className="risk-readout"><div className="risk-number">{(data.risk.max_exceedance_probability * 100).toFixed(0)}<small>%</small></div><div><span>Maximum empirical probability</span><strong>{data.risk.threshold.toFixed(1)} mm threshold</strong></div></div><div className="risk-stats"><div><span>Affected area</span><strong>{metric(data.risk.affected_area_km2, 0)} km²</strong></div><div><span>Severity ratio</span><strong>{metric(data.risk.severity_index, 2)}×</strong></div><div><span>Spread</span><strong>{metric(data.risk.uncertainty, 2)} mm</strong></div></div><p className="limitation-note">Empirical frequency from conditional samples; not a calibrated operational probability.</p></div>}</section>
          </div>

          <section className="panel unavailable-panel"><div className="unavailable-icon" aria-hidden="true">◇</div><div><div className="eyebrow">DOWNSCALING INPUT</div><h2>High-resolution inference</h2><p>The backend requires a coarse precipitation field and coordinates for downscaling and ensemble requests. The selected event response exposes metadata only, so no field is fabricated here.</p></div><span className="soft-badge">Awaiting field input</span></section>
        </section>
      </div>
      <footer className="footer-note">SpatioAI research interface · Synthetic/demo validation mode · Not an official warning system</footer>
    </main>
  )
}

export default App
