import { NavLink, Outlet } from 'react-router-dom'

const tabs = [
  { to: '/', label: 'Home' },
  { to: '/discover', label: 'Discover' },
  { to: '/library', label: 'Library' },
  { to: '/settings', label: 'Settings' }
]

export default function AppShell() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <h1>PulsePlay</h1>
        <p>Emotion-Aware Music App</p>
      </header>
      <main className="page-content">
        <Outlet />
      </main>
      <nav className="tabbar">
        {tabs.map((tab) => (
          <NavLink key={tab.to} to={tab.to} className={({ isActive }) => isActive ? 'tab active' : 'tab'}>
            {tab.label}
          </NavLink>
        ))}
      </nav>
    </div>
  )
}
