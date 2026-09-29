import { useMemo } from 'react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  CartesianGrid,
} from 'recharts'
import { ActivityIcon, WindIcon } from './Icons'
import type { Event, Track } from '../types/api'

type TrackTimelineProps = {
  track: Track | null
  events: Event[]
  selectedEventId: string | null
}

export function TrackTimeline({ track, events, selectedEventId }: TrackTimelineProps) {
  const chartData = useMemo(() => {
    if (!track) return []
    const trackEvents = events.filter((ev) => ev.track_id === track.track_id)
    return trackEvents
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
      .map((ev, idx) => {
        const timeStr = new Date(ev.timestamp).toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
          hour12: false,
        })
        return {
          step: `T+${idx * 6}h (${timeStr})`,
          intensity: Math.round(ev.max_intensity * 10) / 10,
          meanIntensity: Math.round(ev.mean_intensity * 10) / 10,
          area: ev.area_km2,
          eventId: ev.event_id,
          isSelected: ev.event_id === selectedEventId,
        }
      })
  }, [track, events, selectedEventId])

  if (!track || chartData.length === 0) {
    return (
      <section className="card-panel timeline-card" aria-label="Track Dynamics Timeline">
        <div className="panel-eyebrow">TRACK DYNAMICS</div>
        <h3 className="panel-title">Intensity Trajectory Timeline</h3>
        <div className="state-placeholder empty">
          <WindIcon size={20} className="text-slate-500" />
          <span className="text-xs text-slate-400">No multi-timestep storm track associated with the selected anomaly.</span>
        </div>
      </section>
    )
  }

  // Calculate intensity trend
  const firstIntensity = chartData[0]?.intensity ?? 0
  const lastIntensity = chartData[chartData.length - 1]?.intensity ?? 0
  const delta = lastIntensity - firstIntensity

  return (
    <section className="card-panel timeline-card" aria-label="Track Dynamics Timeline">
      <div className="card-header-row">
        <div>
          <span className="panel-eyebrow">KINEMATIC EVOLUTION</span>
          <h3 className="panel-title">Track Intensity Timeline</h3>
        </div>
        <div className="track-chip-badge">
          <span>{track.track_id.replace('TRK_SYNTH_', 'TRK-')}</span>
          <span className="track-steps">{chartData.length} Timesteps (6h Intervals)</span>
        </div>
      </div>

      {/* Chart Summary Stats */}
      <div className="timeline-stat-row">
        <div className="trend-item">
          <ActivityIcon size={12} className={delta >= 0 ? 'text-rose-400' : 'text-sky-400'} />
          <span>Net Delta:</span>
          <strong className={delta >= 0 ? 'text-rose-400 font-mono' : 'text-sky-400 font-mono'}>
            {delta >= 0 ? `+${delta.toFixed(1)}` : delta.toFixed(1)} mm
          </strong>
        </div>
        <div className="trend-item">
          <span>Peak Rate:</span>
          <strong className="text-amber-400 font-mono">{Math.max(...chartData.map((d) => d.intensity))} mm/h</strong>
        </div>
        <div className="trend-item">
          <span>Duration:</span>
          <strong className="font-mono text-slate-300">{(chartData.length - 1) * 6} Hours</strong>
        </div>
      </div>

      {/* Recharts Line Visualization */}
      <div className="timeline-chart-wrap">
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={chartData} margin={{ top: 12, right: 15, left: -20, bottom: 0 }}>
            <CartesianGrid stroke="#1e293b" strokeDasharray="2 2" vertical={false} />
            <XAxis
              dataKey="step"
              stroke="#475569"
              tick={{ fontSize: 10, fill: '#64748b' }}
              tickLine={false}
              axisLine={{ stroke: '#1e293b' }}
            />
            <YAxis
              stroke="#475569"
              tick={{ fontSize: 10, fill: '#64748b' }}
              tickLine={false}
              axisLine={{ stroke: '#1e293b' }}
              unit=" mm"
              domain={['auto', 'auto']}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null
                const d = payload[0].payload
                return (
                  <div className="chart-custom-tooltip">
                    <div className="tooltip-title">{d.step}</div>
                    <div className="tooltip-val text-amber-400 font-mono">
                      Peak Rate: <strong>{d.intensity} mm/h</strong>
                    </div>
                    <div className="tooltip-val text-slate-300 font-mono">
                      Mean Rate: <strong>{d.meanIntensity} mm/h</strong>
                    </div>
                    <div className="tooltip-val text-indigo-300 font-mono">
                      Footprint: <strong>{d.area.toLocaleString()} km&sup2;</strong>
                    </div>
                    {d.isSelected && (
                      <div className="tooltip-badge">Selected Event Step</div>
                    )}
                  </div>
                )
              }}
            />
            <ReferenceLine
              y={100}
              stroke="#ef4444"
              strokeDasharray="3 3"
              label={{ value: '100mm Extreme Threshold', fill: '#f87171', fontSize: 9, position: 'top' }}
            />
            <Line
              type="monotone"
              dataKey="intensity"
              name="Peak Intensity"
              stroke="#f59e0b"
              strokeWidth={2.5}
              dot={{ r: 4, fill: '#f59e0b', stroke: '#080c14', strokeWidth: 1.5 }}
              activeDot={{ r: 6, fill: '#38bdf8', stroke: '#080c14', strokeWidth: 2 }}
            />
            <Line
              type="monotone"
              dataKey="meanIntensity"
              name="Mean Intensity"
              stroke="#38bdf8"
              strokeWidth={1.5}
              strokeDasharray="3 3"
              dot={{ r: 2.5, fill: '#38bdf8' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
