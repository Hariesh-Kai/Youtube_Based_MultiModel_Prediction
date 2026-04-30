import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import AppShell from './layouts/AppShell'
import HomePage from './pages/HomePage'
import DiscoverPage from './pages/DiscoverPage'
import LibraryPage from './pages/LibraryPage'
import SettingsPage from './pages/SettingsPage'
import './styles.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AppShell />}>
          <Route index element={<HomePage />} />
          <Route path="discover" element={<DiscoverPage />} />
          <Route path="library" element={<LibraryPage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
)
