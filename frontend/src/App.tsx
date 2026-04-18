import { useEffect, useState } from 'react'
import './App.css'
import UploadPage from './pages/UploadPage'
import ResultsPage from './pages/ResultsPage'

interface HealthResponse {
  status: string
  app_name: string
  version: string
  timestamp: number
}

type Page =
  | { name: 'home' }
  | { name: 'upload' }
  | { name: 'results'; videoId: string }

function parseHash(): Page {
  const hash = window.location.hash
  if (hash === '#upload') return { name: 'upload' }
  if (hash.startsWith('#results/')) {
    const videoId = hash.slice('#results/'.length)
    if (videoId) return { name: 'results', videoId }
  }
  return { name: 'home' }
}

function App() {
  const [page, setPage] = useState<Page>(parseHash)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const onHash = () => setPage(parseHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch('/api/health')
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data: HealthResponse = await res.json()
        setHealth(data)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Connection failed')
        setHealth(null)
      } finally {
        setLoading(false)
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  if (page.name === 'upload') return <UploadPage />
  if (page.name === 'results') return <ResultsPage videoId={page.videoId} />

  return (
    <div className="app">
      <div className="hero">
        <div className="glow" />
        <h1 className="title">
          <span className="title-multi">Multi</span>
          <span className="title-cheat">Cheat</span>
        </h1>
        <p className="subtitle">AI-Based Exam Cheating Detection System</p>

        <div className="status-card">
          {loading ? (
            <div className="status-row">
              <span className="status-dot pulsing" />
              <span>Connecting to backend…</span>
            </div>
          ) : health ? (
            <>
              <div className="status-row">
                <span className="status-dot online" />
                <span>Backend Online</span>
              </div>
              <div className="status-details">
                <span>v{health.version}</span>
                <span className="divider">•</span>
                <span>{health.app_name}</span>
              </div>
            </>
          ) : (
            <div className="status-row">
              <span className="status-dot offline" />
              <span>Backend Offline — {error}</span>
            </div>
          )}
        </div>

        <div className="features">
          <a href="#upload" className="feature-card feature-card--link">
            <div className="feature-icon">🎥</div>
            <h3>Video Upload</h3>
            <p>Upload exam recordings for analysis</p>
          </a>
          <div className="feature-card">
            <div className="feature-icon">🤖</div>
            <h3>AI Detection</h3>
            <p>Multi-modal cheating behavior detection</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Timeline View</h3>
            <p>Interactive playback with flagged events</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
