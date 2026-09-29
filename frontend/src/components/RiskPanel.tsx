import { ShieldAlertIcon, AlertTriangleIcon, ActivityIcon, SparklesIcon, CpuIcon } from './Icons'
import type { Event, RiskResponse } from '../types/api'

type RiskPanelProps = {
  risk: RiskResponse | null
  loading: boolean
  error: string | null
  currentThreshold: number
  onThresholdChange: (threshold: number) => void
  selectedEvent?: Event | null
}

export function RiskPanel({
  risk,
  loading,
  error,
  currentThreshold,
  onThresholdChange,
  selectedEvent,
}: RiskPanelProps) {
  const probPercent = risk ? Math.round(risk.max_exceedance_probability * 100) : 0
  const isHighRisk = probPercent >= 70
  const isMediumRisk = probPercent >= 40 && probPercent < 70

  const presets = [25, 50, 75, 100, 150]
  const activeThreshold = currentThreshold > 0
    ? currentThreshold
    : selectedEvent
    ? Math.max(25, Math.round(selectedEvent.max_intensity * 0.6))
    : 50

  const isModelUnavailable = Boolean(
    error && (error.toLowerCase().includes('model') || error.toLowerCase().includes('unavailable') || error.toLowerCase().includes('503'))
  )

  // Climatological deterministic baseline when neural weights are offline
  const deterministicSeverity = selectedEvent && activeThreshold > 0
    ? (selectedEvent.max_intensity / activeThreshold).toFixed(2)
    : '1.00'

  return (
    <section className="card-panel risk-card" aria-label="Ensemble Uncertainty & Risk Analysis">
      <div className="card-header-row">
        <div>
          <span className="panel-eyebrow">ENSEMBLE UNCERTAINTY &amp; RISK</span>
          <h3 className="panel-title">Threshold Exceedance Evaluation</h3>
        </div>
        <div className="flex items-center gap-2">
          {risk ? (
            <div className={`risk-level-pill ${isHighRisk ? 'high' : isMediumRisk ? 'medium' : 'low'}`}>
              <ShieldAlertIcon size={13} className="inline mr-1" />
              <span>{isHighRisk ? 'HIGH RISK' : isMediumRisk ? 'MODERATE RISK' : 'LOW RISK'}</span>
            </div>
          ) : (
            <div className="risk-level-pill offline">
              <CpuIcon size={12} className="inline mr-1 text-slate-400" />
              <span>Neural Weights Pending</span>
            </div>
          )}
        </div>
      </div>

      {/* Threshold Selector Tabs */}
      <div className="threshold-selector-strip">
        <span className="selector-title">Exceedance Threshold:</span>
        <div className="preset-pill-group" role="group" aria-label="Exceedance threshold presets">
          {presets.map((val) => (
            <button
              key={val}
              className={`preset-pill ${currentThreshold === val ? 'active' : ''}`}
              onClick={() => onThresholdChange(val)}
              type="button"
            >
              {val} mm
            </button>
          ))}
          <button
            className={`preset-pill ${!presets.includes(currentThreshold) ? 'active' : ''}`}
            onClick={() => onThresholdChange(0)}
            title="Auto: 60% of peak precipitation rate"
            type="button"
          >
            Auto (60% Peak)
          </button>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="state-placeholder">
          <span className="loading-spinner" />
          <span className="text-xs text-slate-400">Sampling conditional generative diffusion ensemble...</span>
        </div>
      )}

      {/* Checkpoint Pending State (Scientific & Authentic) */}
      {!loading && isModelUnavailable && (
        <div className="risk-offline-state">
          <div className="offline-state-header">
            <div className="offline-icon-wrap">
              <AlertTriangleIcon size={18} className="text-amber-400" />
            </div>
            <div>
              <h4 className="offline-title">Diffusion Sampler Offline</h4>
              <p className="offline-subtitle">
                The Phase 8 risk engine requires pre-trained generative diffusion weights (<code>models/diffusion_downscaler_best.pt</code>) to derive empirical ensemble exceedance probabilities.
              </p>
            </div>
          </div>

          {/* Fallback Climatological Baseline Metrics */}
          {selectedEvent && (
            <div className="offline-baseline-box">
              <div className="baseline-header">
                <span>Deterministic Cluster Baseline ({activeThreshold} mm/h Threshold)</span>
                <span className="text-xs text-sky-400 font-mono">EV-{selectedEvent.event_id.replace('EV_SYNTH_', '')}</span>
              </div>

              <div className="baseline-metrics-grid">
                <div className="baseline-metric-item">
                  <span className="bm-label">Peak Rain Rate</span>
                  <span className="bm-val text-amber-400">{selectedEvent.max_intensity.toFixed(1)} <small>mm/h</small></span>
                </div>
                <div className="baseline-metric-item">
                  <span className="bm-label">Threshold Exceedance</span>
                  <span className="bm-val text-rose-400">{deterministicSeverity}&times;</span>
                </div>
                <div className="baseline-metric-item">
                  <span className="bm-label">Cluster Footprint</span>
                  <span className="bm-val">{selectedEvent.area_km2.toLocaleString()} <small>km&sup2;</small></span>
                </div>
                <div className="baseline-metric-item">
                  <span className="bm-label">Mean Rate</span>
                  <span className="bm-val">{selectedEvent.mean_intensity.toFixed(1)} <small>mm/h</small></span>
                </div>
              </div>

              <div className="baseline-train-hint">
                <span className="text-xs text-slate-400">
                  To train the Phase 6 diffusion model weights:
                </span>
                <code className="text-xs text-slate-300 font-mono bg-slate-900 px-2 py-1 rounded block mt-1">
                  python scripts/train_diffusion.py --config configs/data.yaml
                </code>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Live Risk Analysis (When Available) */}
      {!loading && !isModelUnavailable && risk && (
        <div className="risk-content-body">
          <div className="risk-hero-row">
            <div className="probability-display">
              <div className={`probability-number ${isHighRisk ? 'high' : isMediumRisk ? 'medium' : 'low'}`}>
                {probPercent}
                <small>%</small>
              </div>
              <div className="probability-meta">
                <span className="prob-label">Empirical Exceedance</span>
                <span className="prob-sub">
                  Evaluated at <strong>{risk.threshold.toFixed(1)} mm/h</strong> threshold
                </span>
                <div className="prob-progress-bar">
                  <div
                    className={`prob-progress-fill ${isHighRisk ? 'high' : isMediumRisk ? 'medium' : 'low'}`}
                    style={{ width: `${Math.min(100, Math.max(5, probPercent))}%` }}
                  />
                </div>
              </div>
            </div>

            <div className="risk-score-box">
              <span className="score-title">
                <SparklesIcon size={12} className="inline mr-1 text-sky-400" />
                Composite Risk
              </span>
              <span className="score-number">{risk.risk_score.toFixed(2)}</span>
              <span className="score-sub">Severity &times; Footprint Scaled</span>
            </div>
          </div>

          <div className="risk-metrics-grid">
            <div className="risk-metric-box">
              <span className="rm-label">Affected Area</span>
              <span className="rm-val">
                {risk.affected_area_km2.toLocaleString('en-US', { maximumFractionDigits: 0 })} <small>km&sup2;</small>
              </span>
            </div>

            <div className="risk-metric-box">
              <span className="rm-label">Severity Multiplier</span>
              <span className="rm-val text-amber-400">{risk.severity_index.toFixed(2)}&times;</span>
            </div>

            <div className="risk-metric-box">
              <span className="rm-label">Ensemble Spread</span>
              <span className="rm-val text-teal-400">
                &plusmn;{risk.uncertainty.toFixed(2)} <small>mm</small>
              </span>
            </div>

            <div className="risk-metric-box">
              <span className="rm-label">Spatial Mean</span>
              <span className="rm-val">{(risk.mean_exceedance_probability * 100).toFixed(0)}%</span>
            </div>
          </div>

          <div className="scientific-disclaimer">
            <span className="disclaimer-badge">
              <ActivityIcon size={11} className="inline mr-1" />
              DIFFUSION ENSEMBLE
            </span>
            <span>
              Empirical frequency derived from conditional generative diffusion samples; experimental research benchmark, not certified operational weather warning.
            </span>
          </div>
        </div>
      )}
    </section>
  )
}
