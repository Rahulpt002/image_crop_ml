import { NavLink, Outlet } from 'react-router-dom';
import {
  Crop,
  Database,
  GraduationCap,
  FlaskConical,
  Scissors,
  Activity,
} from 'lucide-react';

const navItems = [
  { to: '/trainer', label: 'Trainer', icon: Crop },
  { to: '/dataset', label: 'Dataset', icon: Database },
  { to: '/train', label: 'Train', icon: GraduationCap },
  { to: '/test', label: 'Test', icon: FlaskConical },
  { to: '/crop', label: 'AI Crop', icon: Scissors },
];

export default function Layout() {
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 shrink-0 flex flex-col bg-[var(--color-surface-secondary)] border-r border-[var(--color-border-subtle)]">
        {/* Logo */}
        <div className="px-6 py-5 border-b border-[var(--color-border-subtle)]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-[var(--color-accent-primary)] to-purple-500 flex items-center justify-center">
              <Scissors className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-tight text-[var(--color-text-primary)]">
                AI Crop Trainer
              </h1>
              <p className="text-[10px] text-[var(--color-text-muted)] uppercase tracking-widest">
                Studio
              </p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? 'bg-[var(--color-accent-primary)]/15 text-[var(--color-accent-secondary)] shadow-sm shadow-[var(--color-accent-glow)]'
                    : 'text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-hover)] hover:text-[var(--color-text-primary)]'
                }`
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Status bar */}
        <div className="px-4 py-3 border-t border-[var(--color-border-subtle)]">
          <div className="flex items-center gap-2 text-xs text-[var(--color-text-muted)]">
            <Activity className="w-3 h-3" />
            <span>Local Mode</span>
            <span className="ml-auto inline-block w-2 h-2 rounded-full bg-[var(--color-success)] animate-pulse" />
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto bg-[var(--color-surface-primary)]">
        <Outlet />
      </main>
    </div>
  );
}
