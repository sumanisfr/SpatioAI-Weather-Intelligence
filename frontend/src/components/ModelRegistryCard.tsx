import { CpuIcon, CheckCircleIcon, XCircleIcon, AlertTriangleIcon } from './Icons'
import type { HealthResponse } from '../types/api'

type ModelRegistryCardProps = {
  health: HealthResponse | null
}

const MODEL_METADATA: Record<string, { label: string; desc: string; role: string }> = {
  gnn: {
    label: 'Spatio-Temporal GNN',
    role: 'Phase 4 Trajectory Association',
    desc: 'Deep multi-object graph kinematic trajectory association across variable forecast gaps.',
  },
  unet: {
    label: 'U-Net Downscaler',
    role: 'Phase 5 Statistical Baseline',
    desc: 'Coarse-to-fine statistical resolution baseline downscaling 12 km grid cells to 5 km.',
  },
  diffusion: {
    label: 'Conditional Diffusion',
    role: 'Phase 6 Generative Sampler',
    desc: 'Stochastic conditional score-based diffusion model for precipitation downscaling & ensemble spread.',
  },
  physics_diffusion: {
    label: 'Physics-Informed Diffusion',
    role: 'Phase 7 Regularized Ablation',
    desc: 'Domain-regularized model enforcing integral mass conservation and spatial gradient preservation.',
  },
}

export function ModelRegistryCard({ health }: ModelRegistryCardProps) {
  const models = health?.models ?? {}

  return (
    <section className="card-panel model-registry-card" aria-label="Model Checkpoint Registry">
      <div className="card-header-row">
        <div>
          <span className="panel-eyebrow">INFERENCE ENGINE CHECKPOINTS</span>
          <h3 className="panel-title">Model Registry Status</h3>
        </div>
        <div className="engine-badge">
          <CpuIcon size={13} className="inline mr-1 text-sky-400" />
          <span>FastAPI Neural Engine</span>
        </div>
      </div>

      <div className="models-list">
        {Object.entries(MODEL_METADATA).map(([key, meta]) => {
          const status = models[key]
          const isLoaded = Boolean(status?.loaded)
          const isAvailable = Boolean(status?.available)

          return (
            <div
              key={key}
              className={`model-item-row ${
                isLoaded ? 'ready' : isAvailable ? 'detected' : 'unavailable'
              }`}
            >
              <div className="model-status-indicator">
                {isLoaded ? (
                  <CheckCircleIcon size={15} className="text-emerald-400" />
                ) : isAvailable ? (
                  <AlertTriangleIcon size={15} className="text-amber-400" />
                ) : (
                  <XCircleIcon size={15} className="text-rose-400" />
                )}
              </div>

              <div className="model-info-col">
                <div className="model-name-line">
                  <div className="flex items-center gap-2">
                    <span className="model-name">{meta.label}</span>
                    <span className="model-role-tag">{meta.role}</span>
                  </div>
                  <span
                    className={`status-pill-small ${
                      isLoaded ? 'online' : isAvailable ? 'detected' : 'offline'
                    }`}
                  >
                    {isLoaded
                      ? 'ONLINE'
                      : isAvailable
                      ? 'CHECKPOINT FOUND (UNLOADED)'
                      : 'CHECKPOINT OFFLINE'}
                  </span>
                </div>
                <div className="model-desc">{meta.desc}</div>
                {status?.path && <code className="model-path">{status.path}</code>}
                {status?.error && (
                  <div className="model-error-msg">
                    Diagnostic: {status.error}
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>

      <div className="model-footer-disclaimer">
        <span>
          Models report their actual filesystem status at API boot time. Checkpoints are never silently simulated or substituted in production.
        </span>
      </div>
    </section>
  )
}
