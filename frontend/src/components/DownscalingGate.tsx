import { useState } from 'react'
import { AlertTriangleIcon, ExternalLinkIcon, DatabaseIcon } from './Icons'
import { API_BASE_URL } from '../api/client'

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

  const specUrl = `${API_BASE_URL}/docs#/downscaling/predict_downscaling_api_v1_downscaling_predict_post`

  return (
    <section className="card-panel downscaling-gate-card" aria-label="Input-Gated ML Downscaling">
      {/* Header row */}
      <div className="card-header-row">
        <div>
          <span className="panel-eyebrow">DOWNSCALING &amp; ENSEMBLE PIPELINE</span>
          <h3 className="panel-title">Input-Gated High-Resolution Inference</h3>
        </div>
        <div className="gate-icon-badge">
          <AlertTriangleIcon size={18} className="text-amber-400" />
        </div>
      </div>

      {/* Description */}
      <p className="gate-description">
        The Phase 6 conditional diffusion downscaler and Phase 8 ensemble endpoints require a coarse 2D precipitation
        intensity matrix (12 km grid cells) to generate downscaled 5 km fields. The current event endpoint exposes
        spatio-temporal cluster metadata only — raw precipitation fields are never artificially fabricated in the browser.
      </p>

      {/* Activation info box */}
      <div className="gate-instruction-box">
        <strong className="instruction-heading">
          High-Resolution Downscaling Inference Pipeline
        </strong>
        <span>Requires coarse precipitation intensity tensors from meteorological radar or numerical models.</span>
      </div>

      {/* Action buttons */}
      <div className="gate-actions-row">
        <button
          onClick={() => setShowPayload(!showPayload)}
          type="button"
          className="gate-btn secondary"
        >
          <DatabaseIcon size={12} className="text-sky-400" />
          <span>{showPayload ? 'Hide Payload Schema' : 'Inspect Input Tensor Schema'}</span>
        </button>

        <a
          href={specUrl}
          target="_blank"
          rel="noreferrer"
          className="gate-btn primary"
        >
          <span>Open Downscaling API Spec</span>
          <ExternalLinkIcon size={12} />
        </a>
      </div>

      {/* Payload schema preview */}
      {showPayload && (
        <div className="gate-payload-preview">
          <div className="payload-header">
            <span className="payload-method">POST /api/v1/downscaling/predict</span>
            <code className="payload-mime">application/json</code>
          </div>
          <pre className="payload-body">
            {JSON.stringify(samplePayload, null, 2)}
          </pre>
        </div>
      )}
    </section>
  )
}
