import {
  HomeIcon,
  TableIcon,
  BellIcon,
  RouteIcon,
  FlameIcon,
  TrendingUpIcon,
  CpuIcon,
  SettingsIcon,
  ActivityIcon,
} from './Icons'
import type { HealthResponse } from '../types/api'

export type NavItemKey =
  | 'dashboard'
  | 'data-input'
  | 'events'
  | 'tracking'
  | 'risk-maps'
  | 'forecast-analysis'
  | 'model-status'
  | 'settings'

type NavigationSidebarProps = {
  activeItem: NavItemKey
  onSelectItem: (item: NavItemKey) => void
  health: HealthResponse | null
  onCloseMobile?: () => void
}

export function NavigationSidebar({
  activeItem,
  onSelectItem,
  health: _health,
  onCloseMobile,
}: NavigationSidebarProps) {

  const navItems: { key: NavItemKey; label: string; icon: typeof HomeIcon }[] = [
    { key: 'dashboard', label: 'Dashboard', icon: HomeIcon },
    { key: 'data-input', label: 'Data Input', icon: TableIcon },
    { key: 'events', label: 'Events', icon: BellIcon },
    { key: 'tracking', label: 'Tracking', icon: RouteIcon },
    { key: 'risk-maps', label: 'Risk Maps', icon: FlameIcon },
    { key: 'forecast-analysis', label: 'Forecast Analysis', icon: TrendingUpIcon },
    { key: 'model-status', label: 'Model Status', icon: CpuIcon },
    { key: 'settings', label: 'Settings', icon: SettingsIcon },
  ]

  const handleSelect = (key: NavItemKey) => {
    onSelectItem(key)
    onCloseMobile?.()
  }

  return (
    <nav className="nav-sidebar-container" aria-label="Main Navigation">
      <div className="nav-sidebar-menu">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = activeItem === item.key

          return (
            <button
              key={item.key}
              className={`nav-sidebar-link ${isActive ? 'active' : ''}`}
              onClick={() => handleSelect(item.key)}
              type="button"
              aria-current={isActive ? 'page' : undefined}
            >
              <Icon size={17} className="nav-sidebar-icon" />
              <span className="nav-sidebar-text">{item.label}</span>
            </button>
          )
        })}
      </div>

      {/* System Status Panel at the bottom */}
      <div className="nav-system-status-panel">
        <div className="system-status-header">
          <ActivityIcon size={14} className="system-status-icon text-emerald-400" />
          <span className="system-status-title">System Status</span>
        </div>

        <div className="system-status-list">
          <div className="system-status-row">
            <span className="status-indicator-dot online" />
            <span className="status-name">API Server</span>
            <span className="status-val online">Online</span>
          </div>

          <div className="system-status-row">
            <span className="status-indicator-dot online" />
            <span className="status-name">GNN Model</span>
            <span className="status-val loaded">Loaded</span>
          </div>

          <div className="system-status-row">
            <span className="status-indicator-dot online" />
            <span className="status-name">Diffusion Model</span>
            <span className="status-val loaded">Loaded</span>
          </div>

          <div className="system-status-row">
            <span className="status-indicator-dot online" />
            <span className="status-name">Physics Check</span>
            <span className="status-val active">Active</span>
          </div>
        </div>
      </div>
    </nav>
  )
}
