import { useState } from 'react'
import { DatabaseIcon } from './Icons'

export function DownscalingPanel() {
  const [showPayload, setShowPayload] = useState(false)

  const samplePayload = {
    coarse_precipitation: [
      [12.4, 18.2, 24.5, 30.1],
      [15.8, 38.6, 75.2, 42.0],
      [22.1, 84.9, 112.5, 58.3],
      [10.2, 26.4, 45.1, 21.0],
    ],
    target_resolution_km: 5.0,
    model_type: 'diffusion',
    num_samples: 5,
  }

  // Generate 8x8 coarse block matrix
  const coarseGrid = [
    [0, 1, 1, 2, 2, 1, 0, 0],
    [1, 2, 3, 4, 3, 2, 1, 0],
    [1, 3, 5, 6, 5, 3, 2, 1],
    [2, 4, 6, 7, 6, 4, 3, 1],
    [2, 4, 5, 6, 5, 4, 2, 1],
    [1, 3, 4, 5, 4, 3, 1, 0],
    [0, 2, 3, 3, 2, 1, 0, 0],
    [0, 1, 1, 2, 1, 0, 0, 0],
  ]

  const coarseColors = [
    '#1e293b', // 0
    '#1e3a8a', // 1
    '#2563eb', // 2
    '#06b6d4', // 3
    '#22c55e', // 4
    '#eab308', // 5
    '#f97316', // 6
    '#ef4444', // 7
  ]

  return (
    <section className="downscaling-panel-card" aria-label="High-Resolution Downscaling">
      <div className="panel-header-row">
        <h3 className="panel-header-title">
          High-Resolution Downscaling (12 km &rarr; 5 km)
        </h3>
        <button
          type="button"
          className="panel-header-action-btn"
          onClick={() => setShowPayload(!showPayload)}
          title="Inspect Input Tensor Schema"
        >
          <DatabaseIcon size={12} className="text-sky-400 mr-1" />
          <span>Schema</span>
        </button>
      </div>

      <div className="downscaling-content-row">
        {/* Left: Input (12 km - Coarse) */}
        <div className="downscale-col input-col">
          <div className="downscale-col-title">Input (12 km &ndash; Coarse)</div>
          <div className="coarse-matrix-box">
            <div className="coarse-grid">
              {coarseGrid.map((row, rIdx) => (
                <div key={`r-${rIdx}`} className="coarse-row">
                  {row.map((cell, cIdx) => (
                    <div
                      key={`c-${rIdx}-${cIdx}`}
                      className="coarse-cell"
                      style={{ backgroundColor: coarseColors[cell] }}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Center: Transition Arrow */}
        <div className="downscale-arrow-col">
          <span className="downscale-arrow">&rarr;</span>
        </div>

        {/* Right: Output (5 km - SpatioAI) */}
        <div className="downscale-col output-col">
          <div className="downscale-col-title">Output (5 km &ndash; SpatioAI)</div>
          <div className="fine-swirl-box">
            <svg viewBox="0 0 160 140" className="fine-swirl-svg">
              <defs>
                <radialGradient id="fineGrad" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity="0.95" />
                  <stop offset="25%" stopColor="#f97316" stopOpacity="0.9" />
                  <stop offset="45%" stopColor="#eab308" stopOpacity="0.85" />
                  <stop offset="65%" stopColor="#22c55e" stopOpacity="0.75" />
                  <stop offset="85%" stopColor="#06b6d4" stopOpacity="0.6" />
                  <stop offset="95%" stopColor="#2563eb" stopOpacity="0.4" />
                  <stop offset="100%" stopColor="#1e3a8a" stopOpacity="0.1" />
                </radialGradient>
              </defs>
              <rect width="160" height="140" fill="#0f172a" />
              {/* Outer swirl bands */}
              <path
                d="M 80,70 C 110,40 140,50 155,90 C 145,115 120,125 95,115 C 80,105 75,85 80,70 Z"
                fill="#2563eb"
                opacity="0.4"
              />
              <path
                d="M 80,70 C 50,85 40,115 65,130 C 90,135 115,120 120,95 Z"
                fill="#06b6d4"
                opacity="0.5"
              />
              <ellipse cx="80" cy="70" rx="45" ry="38" fill="url(#fineGrad)" />
              {/* Core eye */}
              <circle cx="80" cy="70" r="8" fill="#990000" />
              <circle cx="80" cy="70" r="3" fill="#ffffff" />
              {/* Trajectory dotted line */}
              <path
                d="M 40,95 Q 60,85 80,70 T 125,45"
                fill="none"
                stroke="#ef4444"
                strokeWidth="1.5"
                strokeDasharray="2,2"
              />
              <circle cx="40" cy="95" r="2" fill="#ef4444" />
              <circle cx="125" cy="45" r="2" fill="#ef4444" />
            </svg>
          </div>
        </div>

        {/* Rightmost: Rainfall Scale */}
        <div className="downscale-scale-col">
          <div className="scale-col-title">Rainfall (mm)</div>
          <div className="scale-col-list">
            <div className="scale-item"><span className="scale-swatch" style={{ backgroundColor: '#990000' }} /><span>&gt; 300</span></div>
            <div className="scale-item"><span className="scale-swatch" style={{ backgroundColor: '#e11d48' }} /><span>200 &ndash; 300</span></div>
            <div className="scale-item"><span className="scale-swatch" style={{ backgroundColor: '#f59e0b' }} /><span>100 &ndash; 200</span></div>
            <div className="scale-item"><span className="scale-swatch" style={{ backgroundColor: '#22c55e' }} /><span>50 &ndash; 100</span></div>
            <div className="scale-item"><span className="scale-swatch" style={{ backgroundColor: '#06b6d4' }} /><span>20 &ndash; 50</span></div>
            <div className="scale-item"><span className="scale-swatch" style={{ backgroundColor: '#2563eb' }} /><span>10 &ndash; 20</span></div>
            <div className="scale-item"><span className="scale-swatch" style={{ backgroundColor: '#1e3a8a' }} /><span>&lt; 10</span></div>
          </div>
        </div>
      </div>

      <div className="panel-footer-caption">
        Preserves extreme rainfall patterns with finer spatial detail
      </div>

      {showPayload && (
        <div className="downscaling-schema-modal">
          <div className="schema-header">
            <span>POST /api/v1/downscaling/predict</span>
            <button
              type="button"
              className="text-xs text-slate-400 hover:text-white"
              onClick={() => setShowPayload(false)}
            >
              &times;
            </button>
          </div>
          <pre className="schema-code">{JSON.stringify(samplePayload, null, 2)}</pre>
        </div>
      )}
    </section>
  )
}
