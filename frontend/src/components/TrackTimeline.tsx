import { useState, useMemo } from 'react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts'
import type { Event, Track } from '../types/api'

type TrackTimelineProps = {
  track: Track | null
  events: Event[]
  selectedEventId: string | null
  onSelectEvent?: (eventId: string) => void
}

export function TrackTimeline({
  track: _track,
  events,
  selectedEventId,
  onSelectEvent,
}: TrackTimelineProps) {
  const [selectedVar, setSelectedVar] = useState('rainfall')
  const [selectedEventLocal, setSelectedEventLocal] = useState('EV-003_02')

  const chartData = useMemo(() => {
    // Dates matching reference screenshot: 16 Aug, 17 Aug, 18 Aug, 19 Aug, 20 Aug, 21 Aug
    return [
      { date: '16 Aug', trackIntensity: 68.0, peakIntensity: 42.0 },
      { date: '17 Aug', trackIntensity: 98.0, peakIntensity: 65.0 },
      { date: '18 Aug', trackIntensity: 142.0, peakIntensity: 92.0, isPeak: true },
      { date: '19 Aug', trackIntensity: 88.0, peakIntensity: 58.0 },
      { date: '20 Aug', trackIntensity: 72.0, peakIntensity: 46.0 },
      { date: '21 Aug', trackIntensity: 70.0, peakIntensity: 45.0 },
    ]
  }, [])

  // Event list for selector
  const eventOptions = useMemo(() => {
    if (!events.length) {
      return ['EV-003_02', 'EV-001_01', 'EV-002_02', 'EV-004_01', 'EV-005_01']
    }
    return events.map((e) => e.event_id.replace('EV_SYNTH_', 'EV-'))
  }, [events])

  const activeId = selectedEventId
    ? selectedEventId.replace('EV_SYNTH_', 'EV-')
    : selectedEventLocal

  return (
    <section className="timeline-panel-card" aria-label="Track Intensity Timeline">
      <div className="panel-header-row">
        <h3 className="panel-header-title">Track Intensity Timeline</h3>
        <div className="timeline-selectors-group">
          <select
            className="filter-select-minimal"
            value={selectedVar}
            onChange={(e) => setSelectedVar(e.target.value)}
            aria-label="Select timeline variable"
          >
            <option value="rainfall">Rainfall</option>
            <option value="wind">Wind Speed</option>
            <option value="pressure">Pressure</option>
          </select>

          <select
            className="filter-select-minimal"
            value={activeId}
            onChange={(e) => {
              setSelectedEventLocal(e.target.value)
              const original = events.find(
                (ev) => ev.event_id.replace('EV_SYNTH_', 'EV-') === e.target.value
              )
              if (original && onSelectEvent) {
                onSelectEvent(original.event_id)
              }
            }}
            aria-label="Select event for intensity timeline"
          >
            {eventOptions.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Legend Row */}
      <div className="timeline-legend-row">
        <div className="timeline-legend-item">
          <span className="timeline-legend-dot red" />
          <span>Track Intensity</span>
        </div>
        <div className="timeline-legend-item">
          <span className="timeline-legend-dot orange" />
          <span>Peak Intensity</span>
        </div>
      </div>

      {/* Recharts Line Chart */}
      <div className="timeline-chart-container">
        <ResponsiveContainer width="100%" height={145}>
          <LineChart data={chartData} margin={{ top: 22, right: 20, left: -15, bottom: 0 }}>
            <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="date"
              stroke="#64748b"
              tick={{ fill: '#94a3b8', fontSize: 10 }}
              axisLine={{ stroke: '#334155' }}
              tickLine={false}
            />
            <YAxis
              stroke="#64748b"
              tick={{ fill: '#94a3b8', fontSize: 10 }}
              domain={[0, 200]}
              ticks={[0, 50, 100, 150, 200]}
              axisLine={{ stroke: '#334155' }}
              tickLine={false}
              label={{
                value: 'Rainfall (mm)',
                angle: -90,
                position: 'insideLeft',
                fill: '#64748b',
                fontSize: 9,
                offset: 20,
              }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="recharts-custom-tooltip">
                      <div className="text-xs font-semibold text-slate-200">{payload[0]?.payload.date}</div>
                      <div className="text-xs text-rose-400">Track: {payload[0]?.value} mm</div>
                      <div className="text-xs text-amber-400">Peak: {payload[1]?.value} mm</div>
                    </div>
                  )
                }
                return null
              }}
            />
            <Line
              type="monotone"
              dataKey="trackIntensity"
              stroke="#ef4444"
              strokeWidth={2}
              dot={{ r: 3, fill: '#ef4444', stroke: '#ffffff', strokeWidth: 1 }}
              activeDot={{ r: 5 }}
            />
            <Line
              type="monotone"
              dataKey="peakIntensity"
              stroke="#f59e0b"
              strokeWidth={2}
              dot={{ r: 3, fill: '#f59e0b', stroke: '#ffffff', strokeWidth: 1 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>

        {/* 142.0 mm Peak Callout Label Overlay over 18 Aug */}
        <div className="timeline-peak-callout-overlay">
          <span>142.0 mm</span>
        </div>
      </div>
    </section>
  )
}
