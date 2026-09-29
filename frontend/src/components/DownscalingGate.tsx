import { useState } from 'react'
import { AlertTriangleIcon, ExternalLinkIcon, DatabaseIcon } from './Icons'

export function DownscalingGate() {
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

  return (
    <section className="card-panel" aria-label="Input-Gated ML Downscaling">
      {/* Header row */}
      <div className="card-header-row">
        <div>
          <span className="panel-eyebrow">DOWNSCALING &amp; ENSEMBLE PIPELINE</span>
          <h3 className="panel-title">Input-Gated High-Resolution Inference</h3>
        </div>
        <div style={{
          width: '36px', height: '36px',
          background: 'rgba(234,179,8,0.1)',
          border: '1px solid rgba(234,179,8,0.3)',
          borderRadius: '10px',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 0 16px rgba(234,179,8,0.1)',
          flexShrink: 0,
        }}>
          <AlertTriangleIcon size={18} className="text-amber-400" />
        </div>
      </div>

      {/* Description */}
      <p style={{ fontSize: '12px', color: '#94a3b8', lineHeight: 1.6, marginBottom: '14px' }}>
        The Phase 6 conditional diffusion downscaler and Phase 8 ensemble endpoints require a coarse 2D precipitation
        intensity matrix (12 km grid cells) to generate downscaled 5 km fields. The current event endpoint exposes
        spatio-temporal cluster metadata only — no raw precipitation fields are artificially fabricated in the browser.
      </p>

      {/* Info box */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(6,182,212,0.06), rgba(139,92,246,0.05))',
        border: '1px solid rgba(6,182,212,0.2)',
        borderRadius: '10px',
        padding: '10px 12px',
        marginBottom: '14px',
        fontSize: '11px',
        color: '#94a3b8',
        lineHeight: 1.6,
      }}>
        <strong style={{ color: '#22d3ee', display: 'block', marginBottom: '4px', fontSize: '11px' }}>
          To activate full downscaling inference:
        </strong>
        Train the Phase 6 diffusion model by running:
        <code style={{
          display: 'block', marginTop: '6px',
          background: 'rgba(3,7,18,0.6)',
          border: '1px solid rgba(255,255,255,0.07)',
          borderRadius: '6px',
          padding: '6px 10px',
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: '11px',
          color: '#a78bfa',
        }}>
          python scripts/train_diffusion.py --config configs/data.yaml
        </code>
      </div>

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        <button
          onClick={() => setShowPayload(!showPayload)}
          type="button"
          style={{
            display: 'flex', alignItems: 'center', gap: '6px',
            padding: '7px 12px',
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '8px',
            color: '#94a3b8',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.18s',
            fontFamily: "'Inter', sans-serif",
          }}
          onMouseEnter={e => {
            (e.currentTarget as HTMLElement).style.background = 'rgba(6,182,212,0.1)'
            ;(e.currentTarget as HTMLElement).style.borderColor = 'rgba(6,182,212,0.3)'
            ;(e.currentTarget as HTMLElement).style.color = '#22d3ee'
          }}
          onMouseLeave={e => {
            (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.04)'
            ;(e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.1)'
            ;(e.currentTarget as HTMLElement).style.color = '#94a3b8'
          }}
        >
          <DatabaseIcon size={12} className="text-cyan-400" />
          {showPayload ? 'Hide Payload Schema' : 'Inspect Input Tensor Schema'}
        </button>

        <a
          href="http://localhost:8000/docs#/downscaling/predict_downscaling_api_v1_downscaling_predict_post"
          target="_blank"
          rel="noreferrer"
          style={{
            display: 'flex', alignItems: 'center', gap: '6px',
            padding: '7px 12px',
            background: 'linear-gradient(135deg, rgba(139,92,246,0.15), rgba(6,182,212,0.12))',
            border: '1px solid rgba(139,92,246,0.3)',
            borderRadius: '8px',
            color: '#a78bfa',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.18s',
            textDecoration: 'none',
          }}
          onMouseEnter={e => { (e.currentTarget as HTMLElement).style.boxShadow = '0 0 14px rgba(139,92,246,0.2)' }}
          onMouseLeave={e => { (e.currentTarget as HTMLElement).style.boxShadow = 'none' }}
        >
          Open Downscaling API Spec
          <ExternalLinkIcon size={12} />
        </a>
      </div>

      {/* Payload schema */}
      {showPayload && (
        <div style={{
          marginTop: '12px',
          background: 'rgba(3,7,18,0.7)',
          border: '1px solid rgba(139,92,246,0.2)',
          borderRadius: '10px',
          overflow: 'hidden',
          animation: 'fade-in-up 0.2s ease',
        }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '8px 12px',
            borderBottom: '1px solid rgba(255,255,255,0.05)',
            background: 'rgba(139,92,246,0.08)',
          }}>
            <span style={{ fontSize: '10px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              POST /api/v1/downscaling/predict
            </span>
            <code style={{ fontSize: '9px', color: '#475569', background: 'rgba(255,255,255,0.04)', padding: '2px 6px', borderRadius: '4px' }}>
              application/json
            </code>
          </div>
          <pre style={{
            margin: 0, padding: '12px',
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: '11px',
            color: '#a78bfa',
            lineHeight: 1.6,
            overflowX: 'auto',
          }}>
            {JSON.stringify(samplePayload, null, 2)}
          </pre>
        </div>
      )}
    </section>
  )
}
