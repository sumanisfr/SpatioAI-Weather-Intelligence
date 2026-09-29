import { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertTriangleIcon } from './Icons'

type Props = {
  children: ReactNode
  fallbackTitle?: string
}

type State = {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  }

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('SpatioAI UI Uncaught Exception:', error, errorInfo)
  }

  public handleReset = () => {
    this.setState({ hasError: false, error: null })
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="card-panel error-boundary-panel" role="alert">
          <div className="flex items-center gap-3">
            <AlertTriangleIcon size={20} className="text-amber-400 shrink-0" />
            <div>
              <h4 className="text-sm font-semibold text-slate-200">
                {this.props.fallbackTitle ?? 'Component Display Suspended'}
              </h4>
              <p className="text-xs text-slate-400 mt-0.5">
                {this.state.error?.message ?? 'A non-fatal rendering error occurred in this workspace block.'}
              </p>
            </div>
          </div>
          <div className="mt-3">
            <button
              type="button"
              onClick={this.handleReset}
              className="action-icon-btn text-xs px-2.5 py-1 text-slate-300 hover:text-white"
            >
              Retry Render
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
