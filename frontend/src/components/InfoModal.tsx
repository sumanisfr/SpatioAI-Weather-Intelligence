import { CloseIcon, InfoIcon, ShieldAlertIcon, RadarIcon, ActivityIcon } from './Icons'

type InfoModalProps = {
  isOpen: boolean
  onClose: () => void
}

export function InfoModal({ isOpen, onClose }: InfoModalProps) {
  if (!isOpen) return null

  return (
    <div
      className="info-modal-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
    >
      <div className="info-modal-panel" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
          padding: '22px 24px 18px',
          borderBottom: '1px solid rgba(139,92,246,0.2)',
          background: 'linear-gradient(135deg, rgba(139,92,246,0.08) 0%, rgba(6,182,212,0.05) 100%)',
          borderRadius: '20px 20px 0 0',
        }}>
          <div style={{ display: 'flex', gap: '14px', alignItems: 'center' }}>
            <div style={{
              width: '44px', height: '44px',
              background: 'linear-gradient(135deg, rgba(6,182,212,0.2), rgba(139,92,246,0.2))',
              border: '1px solid rgba(139,92,246,0.4)',
              borderRadius: '12px',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 0 20px rgba(139,92,246,0.2)',
            }}>
              <RadarIcon size={22} className="text-cyan-400" />
            </div>
            <div>
              <div style={{ fontSize: '9px', fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase',
                background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text', marginBottom: '3px' }}>
                RESEARCH ARCHITECTURE
              </div>
              <h3 id="modal-title" style={{
                fontFamily: "'Space Grotesk', sans-serif", fontSize: '18px', fontWeight: 800,
                color: '#f1f5f9', margin: 0, letterSpacing: '-0.01em'
              }}>
                SpatioAI Scientific Scope &amp; Pipeline
              </h3>
              <p style={{ fontSize: '11px', color: '#64748b', margin: '4px 0 0', fontStyle: 'italic' }}>
                Smart India Hackathon (SIH) Meteorological Research Prototype
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close dialog"
            type="button"
            style={{
              background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '8px', padding: '6px', cursor: 'pointer',
              color: '#94a3b8', display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'all 0.18s',
            }}
            onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(239,68,68,0.15)'; (e.currentTarget as HTMLElement).style.color = '#f87171' }}
            onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.06)'; (e.currentTarget as HTMLElement).style.color = '#94a3b8' }}
          >
            <CloseIcon size={18} />
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Section 1 */}
          <div style={{
            background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(139,92,246,0.15)',
            borderRadius: '14px', padding: '16px',
          }}>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginBottom: '12px' }}>
              <ShieldAlertIcon size={16} className="text-amber-400" />
              <h4 style={{ fontFamily: "'Space Grotesk',sans-serif", fontSize: '13px', fontWeight: 700, color: '#f1f5f9', margin: 0 }}>
                Scientific Scope &amp; Operational Distinctions
              </h4>
            </div>
            <p style={{ fontSize: '12px', color: '#94a3b8', margin: '0 0 12px', lineHeight: 1.6 }}>
              SpatioAI is an experimental meteorological intelligence framework benchmarking AI-driven spatio-temporal tracking,
              generative diffusion downscaling, and empirical threshold-exceedance risk quantification for extreme precipitation anomalies over India and the Bay of Bengal basin.
            </p>
            <ul style={{ margin: 0, padding: '0 0 0 16px', display: 'flex', flexDirection: 'column', gap: '7px', fontSize: '11px', color: '#94a3b8', lineHeight: 1.5 }}>
              {[
                ['Reanalysis (ERA5)', 'Historical blended model-observation numerical baseline, not direct ground-truth station observations.'],
                ['NWP (NCUM / NEPS-G / GFS)', 'Deterministic and ensemble physics-based numerical simulations for medium-range forecasts.'],
                ['Hungarian Tracker Baseline', 'Classical deterministic multi-object trajectory association using geodesic distances and Hungarian matching.'],
                ['Generative Diffusion Downscaling', 'Conditional score-based diffusion model mapping coarse 12 km fields to fine 5 km fields with empirical uncertainty.'],
                ['Research Prototype', 'Experimental research workstation — not an operational weather agency service or certified warning broadcast.'],
              ].map(([key, val]) => (
                <li key={key}><strong style={{ color: '#f1f5f9' }}>{key}:</strong> {val}</li>
              ))}
            </ul>
          </div>

          {/* Section 2: Pipeline Phases */}
          <div>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginBottom: '12px' }}>
              <InfoIcon size={16} className="text-cyan-400" />
              <h4 style={{ fontFamily: "'Space Grotesk',sans-serif", fontSize: '13px', fontWeight: 700, color: '#f1f5f9', margin: 0 }}>
                Implemented End-to-End Pipeline Phases
              </h4>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              {[
                ['Phase 1: Ingestion & Regridding', 'ERA5 & NWP loaders, coordinate standardization, 12 km grid regridding, Zarr chunking.'],
                ['Phase 2: Climatology & Anomaly Engine', 'Time-aware climatological baselines, z-score thresholding, connected components segmentation.'],
                ['Phase 3: Hungarian Tracking', 'Haversine geodesics, bearing angles, IoU costs, Hungarian assignment, Kalman gap bridging.'],
                ['Phase 6 & 7: Conditional Diffusion', '12 km → 5 km generative downscaling, log1p transform, differentiable physics constraints.'],
                ['Phase 8: Ensemble Risk', 'Empirical threshold-exceedance probabilities, spatial scaling, transparent risk score computation.'],
                ['Phase 9 & 10: FastAPI & Dashboard', 'Versioned REST API, Leaflet GIS visualization, track analytics, live station telemetry.'],
              ].map(([title, desc]) => (
                <div key={title} style={{
                  background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)',
                  borderRadius: '10px', padding: '10px 12px',
                }}>
                  <strong style={{ display: 'block', fontSize: '11px', color: '#a78bfa', marginBottom: '4px', fontFamily: "'Space Grotesk',sans-serif" }}>{title}</strong>
                  <span style={{ fontSize: '11px', color: '#64748b', lineHeight: 1.5 }}>{desc}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Section 3: Flow */}
          <div>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginBottom: '12px' }}>
              <ActivityIcon size={16} className="text-teal-400" />
              <h4 style={{ fontFamily: "'Space Grotesk',sans-serif", fontSize: '13px', fontWeight: 700, color: '#f1f5f9', margin: 0 }}>
                Operational Data Flow
              </h4>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              {[
                ['01', 'Coarse Grid Ingestion', '12 km numerical precipitation fields from reanalysis / NWP forecast ensembles.'],
                ['02', 'Anomaly Segmentation', 'Thresholding at >95th climatological percentile to isolate spatial candidate clusters.'],
                ['03', 'Kinematic Association', 'Multi-factor Hungarian matching links centroids across 6-hour timesteps into coherent tracks.'],
                ['04', 'Diffusion Downscaling & Risk', 'Conditional U-Net samples fine 5 km fields, derives empirical threshold exceedance zones.'],
              ].map(([num, title, desc]) => (
                <div key={num} style={{
                  background: 'linear-gradient(135deg, rgba(6,182,212,0.05), rgba(139,92,246,0.04))',
                  border: '1px solid rgba(6,182,212,0.15)',
                  borderRadius: '10px', padding: '12px',
                  display: 'flex', gap: '12px', alignItems: 'flex-start',
                }}>
                  <span style={{
                    fontFamily: "'JetBrains Mono', monospace", fontSize: '18px', fontWeight: 800,
                    background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
                    WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
                    lineHeight: 1, flexShrink: 0,
                  }}>{num}</span>
                  <div>
                    <strong style={{ display: 'block', fontSize: '11px', color: '#f1f5f9', marginBottom: '3px', fontFamily: "'Space Grotesk',sans-serif" }}>{title}</strong>
                    <span style={{ fontSize: '11px', color: '#64748b', lineHeight: 1.5 }}>{desc}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: '14px 24px',
          borderTop: '1px solid rgba(139,92,246,0.15)',
          background: 'rgba(255,255,255,0.02)',
          borderRadius: '0 0 20px 20px',
        }}>
          <span style={{ fontSize: '10px', color: '#475569' }}>
            SpatioAI Meteorological Intelligence Prototype &bull; Indian Subcontinent Basin (8°N–28°N, 68°E–94°E)
          </span>
          <button
            onClick={onClose}
            type="button"
            style={{
              padding: '7px 16px',
              background: 'linear-gradient(135deg, rgba(139,92,246,0.3), rgba(6,182,212,0.25))',
              border: '1px solid rgba(6,182,212,0.4)',
              borderRadius: '8px',
              color: '#e2e8f0',
              fontSize: '11px',
              fontWeight: 700,
              cursor: 'pointer',
              transition: 'all 0.18s',
              fontFamily: "'Inter', sans-serif",
            }}
            onMouseEnter={e => { (e.currentTarget as HTMLElement).style.boxShadow = '0 0 20px rgba(6,182,212,0.3)' }}
            onMouseLeave={e => { (e.currentTarget as HTMLElement).style.boxShadow = 'none' }}
          >
            Close Overview
          </button>
        </div>
      </div>
    </div>
  )
}
