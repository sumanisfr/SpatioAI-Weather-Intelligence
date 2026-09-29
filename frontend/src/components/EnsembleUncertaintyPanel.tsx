import { useState } from 'react'

type EnsembleMode = 'mean' | 'std' | 'threshold'

type EnsembleUncertaintyPanelProps = {
  currentThreshold?: number
  onThresholdChange?: (val: number) => void
}

export function EnsembleUncertaintyPanel({
  currentThreshold: _currentThreshold = 50,
  onThresholdChange: _onThresholdChange,
}: EnsembleUncertaintyPanelProps) {
  const [activeTab, setActiveTab] = useState<EnsembleMode>('mean')

  return (
    <section className="ensemble-panel-card" aria-label="Ensemble Uncertainty">
      <div className="panel-header-row">
        <h3 className="panel-header-title">Ensemble Uncertainty</h3>
      </div>

      {/* Tabs */}
      <div className="ensemble-tabs-row" role="tablist">
        <button
          type="button"
          className={`ensemble-tab-btn ${activeTab === 'mean' ? 'active' : ''}`}
          onClick={() => setActiveTab('mean')}
          role="tab"
          aria-selected={activeTab === 'mean'}
        >
          Mean Prediction
        </button>
        <button
          type="button"
          className={`ensemble-tab-btn ${activeTab === 'std' ? 'active' : ''}`}
          onClick={() => setActiveTab('std')}
          role="tab"
          aria-selected={activeTab === 'std'}
        >
          Uncertainty (Std)
        </button>
        <button
          type="button"
          className={`ensemble-tab-btn ${activeTab === 'threshold' ? 'active' : ''}`}
          onClick={() => setActiveTab('threshold')}
          role="tab"
          aria-selected={activeTab === 'threshold'}
        >
          Probability &gt; Threshold
        </button>
      </div>

      {/* Visual Content: Heatmap + Scale */}
      <div className="ensemble-content-row">
        <div className="ensemble-map-preview">
          <svg viewBox="0 0 200 130" className="ensemble-svg">
            <defs>
              <radialGradient id="ensGrad" cx="45%" cy="50%" r="50%">
                <stop offset="0%" stopColor={activeTab === 'std' ? '#a855f7' : '#ef4444'} stopOpacity="0.95" />
                <stop offset="25%" stopColor={activeTab === 'std' ? '#818cf8' : '#f97316'} stopOpacity="0.85" />
                <stop offset="50%" stopColor={activeTab === 'std' ? '#38bdf8' : '#eab308'} stopOpacity="0.75" />
                <stop offset="70%" stopColor={activeTab === 'std' ? '#34d399' : '#22c55e'} stopOpacity="0.6" />
                <stop offset="85%" stopColor={activeTab === 'std' ? '#06b6d4' : '#06b6d4'} stopOpacity="0.4" />
                <stop offset="100%" stopColor="#0f172a" stopOpacity="0.0" />
              </radialGradient>
            </defs>

            <rect width="200" height="130" fill="#0f172a" />
            {/* Background land hints */}
            <path d="M 0,40 Q 50,30 80,60 T 120,110 L 120,130 L 0,130 Z" fill="#1e293b" opacity="0.6" />

            {/* Heatmap spread */}
            <ellipse cx="90" cy="65" rx="55" ry="40" fill="url(#ensGrad)" />
            {activeTab !== 'std' && (
              <>
                <circle cx="90" cy="65" r="9" fill="#dc2626" />
                <circle cx="90" cy="65" r="3" fill="#ffffff" />
              </>
            )}

            {/* Storm trajectory line with points */}
            <path
              d="M 45,90 Q 70,80 90,65 T 145,35"
              fill="none"
              stroke="#ef4444"
              strokeWidth="2"
            />
            <circle cx="45" cy="90" r="3" fill="#ffffff" stroke="#ef4444" strokeWidth="1.5" />
            <circle cx="70" cy="78" r="3" fill="#ffffff" stroke="#ef4444" strokeWidth="1.5" />
            <circle cx="90" cy="65" r="4" fill="#ef4444" stroke="#ffffff" strokeWidth="1.5" />
            <circle cx="115" cy="50" r="3" fill="#ffffff" stroke="#ef4444" strokeWidth="1.5" />
            <circle cx="145" cy="35" r="3" fill="#ffffff" stroke="#ef4444" strokeWidth="1.5" />
          </svg>
        </div>

        {/* Probability (%) Scale */}
        <div className="ensemble-scale-col">
          <div className="scale-col-title">Probability (%)</div>
          <div className="scale-col-list">
            <div className="scale-item"><span className="scale-swatch" style={{ backgroundColor: '#ef4444' }} /><span>&gt; 80</span></div>
            <div className="scale-item"><span className="scale-swatch" style={{ backgroundColor: '#f97316' }} /><span>60 &ndash; 80</span></div>
            <div className="scale-item"><span className="scale-swatch" style={{ backgroundColor: '#eab308' }} /><span>40 &ndash; 60</span></div>
            <div className="scale-item"><span className="scale-swatch" style={{ backgroundColor: '#22c55e' }} /><span>20 &ndash; 40</span></div>
            <div className="scale-item"><span className="scale-swatch" style={{ backgroundColor: '#06b6d4' }} /><span>&lt; 20</span></div>
          </div>
        </div>
      </div>

      <div className="panel-footer-caption">
        Multiple predictions to estimate uncertainty and risk
      </div>
    </section>
  )
}
