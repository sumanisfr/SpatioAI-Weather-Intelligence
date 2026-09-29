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
        {/* Modal Header */}
        <div className="modal-header-strip">
          <div className="modal-header-left">
            <div className="modal-icon-badge">
              <RadarIcon size={20} className="text-sky-400" />
            </div>
            <div>
              <span className="modal-eyebrow">RESEARCH ARCHITECTURE</span>
              <h3 id="modal-title" className="modal-title">
                Sanket Scientific Scope &amp; Pipeline
              </h3>
              <p className="modal-subtitle">
                Smart India Hackathon (SIH) Meteorological Research Prototype
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close dialog"
            type="button"
            className="modal-close-btn"
          >
            <CloseIcon size={16} />
          </button>
        </div>

        {/* Modal Body */}
        <div className="modal-body-content">
          {/* Section 1: Scientific Scope */}
          <div className="modal-section-card">
            <div className="modal-section-header">
              <ShieldAlertIcon size={16} className="text-amber-400" />
              <h4>Scientific Scope &amp; Operational Distinctions</h4>
            </div>
            <p className="modal-section-text">
              Sanket is an experimental meteorological intelligence framework benchmarking AI-driven spatio-temporal tracking,
              generative diffusion downscaling, and empirical threshold-exceedance risk quantification for extreme precipitation anomalies over India and the Bay of Bengal basin.
            </p>
            <ul className="modal-bullets-list">
              {[
                ['Reanalysis (ERA5)', 'Historical blended model-observation numerical baseline, not direct ground-truth station observations.'],
                ['NWP (NCUM / NEPS-G / GFS)', 'Deterministic and ensemble physics-based numerical simulations for medium-range forecasts.'],
                ['Hungarian Tracker Baseline', 'Classical deterministic multi-object trajectory association using geodesic distances and Hungarian matching.'],
                ['Generative Diffusion Downscaling', 'Conditional score-based diffusion model mapping coarse 12 km fields to fine 5 km fields with empirical uncertainty.'],
                ['Research Prototype', 'Experimental research workstation — not an operational weather agency service or certified warning broadcast.'],
              ].map(([key, val]) => (
                <li key={key}><strong>{key}:</strong> {val}</li>
              ))}
            </ul>
          </div>

          {/* Section 2: Implemented Pipeline Phases */}
          <div className="modal-section-card">
            <div className="modal-section-header">
              <InfoIcon size={16} className="text-sky-400" />
              <h4>Implemented End-to-End Pipeline Phases</h4>
            </div>
            <div className="modal-grid-phases">
              {[
                ['Phase 1: Ingestion & Regridding', 'ERA5 & NWP loaders, coordinate standardization, 12 km grid regridding, Zarr chunking.'],
                ['Phase 2: Climatology & Anomaly Engine', 'Time-aware climatological baselines, z-score thresholding, connected components segmentation.'],
                ['Phase 3: Hungarian Tracking', 'Haversine geodesics, bearing angles, IoU costs, Hungarian assignment, Kalman gap bridging.'],
                ['Phase 6 & 7: Conditional Diffusion', '12 km → 5 km generative downscaling, log1p transform, differentiable physics constraints.'],
                ['Phase 8: Ensemble Risk', 'Empirical threshold-exceedance probabilities, spatial scaling, transparent risk score computation.'],
                ['Phase 9 & 10: FastAPI & Dashboard', 'Versioned REST API, Leaflet GIS visualization, track analytics, live station telemetry.'],
              ].map(([title, desc]) => (
                <div key={title} className="phase-card-item">
                  <strong>{title}</strong>
                  <span>{desc}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Section 3: Data Flow */}
          <div className="modal-section-card">
            <div className="modal-section-header">
              <ActivityIcon size={16} className="text-teal-400" />
              <h4>Operational Data Flow</h4>
            </div>
            <div className="modal-flow-grid">
              {[
                ['01', 'Coarse Grid Ingestion', '12 km numerical precipitation fields from reanalysis / NWP forecast ensembles.'],
                ['02', 'Anomaly Segmentation', 'Thresholding at >95th climatological percentile to isolate spatial candidate clusters.'],
                ['03', 'Kinematic Association', 'Multi-factor Hungarian matching links centroids across 6-hour timesteps into coherent tracks.'],
                ['04', 'Diffusion Downscaling & Risk', 'Conditional U-Net samples fine 5 km fields, derives empirical threshold exceedance zones.'],
              ].map(([num, title, desc]) => (
                <div key={num} className="flow-step-item">
                  <span className="flow-step-number">{num}</span>
                  <div>
                    <strong>{title}</strong>
                    <span>{desc}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="modal-footer-strip">
          <span className="modal-basin-note">
            Sanket Meteorological Intelligence Console &bull; Indian Subcontinent Basin (8&deg;N&ndash;28&deg;N, 68&deg;E&ndash;94&deg;E)
          </span>
          <button
            onClick={onClose}
            type="button"
            className="modal-footer-btn"
          >
            Close Overview
          </button>
        </div>
      </div>
    </div>
  )
}
