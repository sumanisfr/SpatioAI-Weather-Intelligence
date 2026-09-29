import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  LabelList,
} from 'recharts'

export function ModelBenchmarkRow() {
  // Chart 1: Model Performance
  const modelPerfData = [
    { metric: 'MAE (mm)', baseline: 42.5, spatio: 18.7 },
    { metric: 'RMSE (mm)', baseline: 48.2, spatio: 21.6 },
  ]

  // Chart 2: Extreme Value Preservation
  const extremeValData = [
    { metric: '95th Percentile', baseline: 68.4, spatio: 22.1 },
    { metric: '99th Percentile', baseline: 92.3, spatio: 28.7 },
    { metric: 'Peak Intensity', baseline: 76.5, spatio: 19.8 },
  ]

  // Chart 3: Event Tracking Performance
  const trackingPerfData = [
    { metric: 'Precision', baseline: 0.62, gnn: 0.86 },
    { metric: 'Recall', baseline: 0.58, gnn: 0.83 },
    { metric: 'F1-Score', baseline: 0.60, gnn: 0.85 },
    { metric: 'PR-AUC', baseline: 0.66, gnn: 0.88 },
  ]

  return (
    <div className="benchmark-panels-grid" aria-label="Model Benchmark Performance">
      {/* Panel 1: Model Performance */}
      <section className="benchmark-card" aria-label="Model Performance">
        <div className="benchmark-header">
          <h3 className="benchmark-title">Model Performance (Before vs Sanket)</h3>
          <div className="benchmark-legend">
            <span className="legend-item">
              <span className="legend-box" style={{ backgroundColor: '#38bdf8' }} />
              Baseline (12 km)
            </span>
            <span className="legend-item">
              <span className="legend-box" style={{ backgroundColor: '#a855f7' }} />
              Sanket (5 km)
            </span>
          </div>
        </div>

        <div className="benchmark-chart-wrap">
          <ResponsiveContainer width="100%" height={140}>
            <BarChart data={modelPerfData} margin={{ top: 18, right: 10, left: -25, bottom: 0 }}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="metric"
                stroke="#64748b"
                tick={{ fill: '#94a3b8', fontSize: 10 }}
                axisLine={{ stroke: '#334155' }}
                tickLine={false}
              />
              <YAxis
                stroke="#64748b"
                tick={{ fill: '#94a3b8', fontSize: 9 }}
                domain={[0, 55]}
                ticks={[0, 10, 20, 30, 40, 50]}
                axisLine={{ stroke: '#334155' }}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 4, fontSize: 11 }}
              />
              <Bar dataKey="baseline" fill="#38bdf8" radius={[2, 2, 0, 0]} maxBarSize={28}>
                <LabelList dataKey="baseline" position="top" fill="#cbd5e1" fontSize={9} />
              </Bar>
              <Bar dataKey="spatio" fill="#a855f7" radius={[2, 2, 0, 0]} maxBarSize={28}>
                <LabelList dataKey="spatio" position="top" fill="#cbd5e1" fontSize={9} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="benchmark-caption">
          Lower error indicates improved reconstruction of fine-scale precipitation.
        </div>
      </section>

      {/* Panel 2: Extreme Value Preservation */}
      <section className="benchmark-card" aria-label="Extreme Value Preservation">
        <div className="benchmark-header">
          <h3 className="benchmark-title">Extreme Value Preservation</h3>
          <div className="benchmark-legend">
            <span className="legend-item">
              <span className="legend-box" style={{ backgroundColor: '#38bdf8' }} />
              Baseline (12 km)
            </span>
            <span className="legend-item">
              <span className="legend-box" style={{ backgroundColor: '#a855f7' }} />
              Sanket (5 km)
            </span>
          </div>
        </div>

        <div className="benchmark-chart-wrap">
          <ResponsiveContainer width="100%" height={140}>
            <BarChart data={extremeValData} margin={{ top: 18, right: 10, left: -25, bottom: 0 }}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="metric"
                stroke="#64748b"
                tick={{ fill: '#94a3b8', fontSize: 10 }}
                axisLine={{ stroke: '#334155' }}
                tickLine={false}
              />
              <YAxis
                stroke="#64748b"
                tick={{ fill: '#94a3b8', fontSize: 9 }}
                domain={[0, 100]}
                ticks={[0, 20, 40, 60, 80, 100]}
                axisLine={{ stroke: '#334155' }}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 4, fontSize: 11 }}
              />
              <Bar dataKey="baseline" fill="#38bdf8" radius={[2, 2, 0, 0]} maxBarSize={28}>
                <LabelList dataKey="baseline" position="top" fill="#cbd5e1" fontSize={9} />
              </Bar>
              <Bar dataKey="spatio" fill="#a855f7" radius={[2, 2, 0, 0]} maxBarSize={28}>
                <LabelList dataKey="spatio" position="top" fill="#cbd5e1" fontSize={9} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="benchmark-caption">
          Preserves high-intensity rainfall patterns during downscaling.
        </div>
      </section>

      {/* Panel 3: Event Tracking Performance (GNN) */}
      <section className="benchmark-card" aria-label="Event Tracking Performance">
        <div className="benchmark-header">
          <h3 className="benchmark-title">Event Tracking Performance (GNN)</h3>
          <div className="benchmark-legend">
            <span className="legend-item">
              <span className="legend-box" style={{ backgroundColor: '#38bdf8' }} />
              Baseline (Geospatial)
            </span>
            <span className="legend-item">
              <span className="legend-box" style={{ backgroundColor: '#a855f7' }} />
              GNN (Sanket)
            </span>
          </div>
        </div>

        <div className="benchmark-chart-wrap">
          <ResponsiveContainer width="100%" height={140}>
            <BarChart data={trackingPerfData} margin={{ top: 18, right: 10, left: -25, bottom: 0 }}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="metric"
                stroke="#64748b"
                tick={{ fill: '#94a3b8', fontSize: 10 }}
                axisLine={{ stroke: '#334155' }}
                tickLine={false}
              />
              <YAxis
                stroke="#64748b"
                tick={{ fill: '#94a3b8', fontSize: 9 }}
                domain={[0, 1.0]}
                ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
                axisLine={{ stroke: '#334155' }}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 4, fontSize: 11 }}
              />
              <Bar dataKey="baseline" fill="#38bdf8" radius={[2, 2, 0, 0]} maxBarSize={22}>
                <LabelList dataKey="baseline" position="top" fill="#cbd5e1" fontSize={9} />
              </Bar>
              <Bar dataKey="gnn" fill="#a855f7" radius={[2, 2, 0, 0]} maxBarSize={22}>
                <LabelList dataKey="gnn" position="top" fill="#cbd5e1" fontSize={9} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="benchmark-caption">
          Accurately tracks the location and movement of extreme weather events.
        </div>
      </section>
    </div>
  )
}
