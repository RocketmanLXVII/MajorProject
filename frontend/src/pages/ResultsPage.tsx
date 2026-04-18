import { useState, useEffect } from 'react'
import './ResultsPage.css'
import TimelinePlayer from '../components/TimelinePlayer'

/* ── Types ── */
interface FlagEvent {
  event_id: string
  video_id: string
  start_timestamp: number
  end_timestamp: number
  start_frame: number
  end_frame: number
  predicted_class: string
  confidence_score: number
  severity: string
  explanation_text: string
  evidence_frame_paths: string[]
  annotated_frame_paths: string[]
  model_version: string
}

interface AnalysisResult {
  video_id: string
  events: FlagEvent[]
  total_frames_analyzed: number
  total_duration_seconds: number
  processing_time_seconds: number
  model_version: string
  analysis_timestamp: string
}

interface VideoMeta {
  video_id: string
  original_filename: string
  duration_seconds: number
  fps: number
  width: number
  height: number
  frame_count: number
  status: string
}

interface Props {
  videoId: string
}

const SEVERITY_COLORS: Record<string, string> = {
  low: '#34d399',
  medium: '#fbbf24',
  high: '#f97316',
  critical: '#ef4444',
}

export default function ResultsPage({ videoId }: Props) {
  const [meta, setMeta] = useState<VideoMeta | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState('')
  const [expandedEvent, setExpandedEvent] = useState<string | null>(null)
  const [lightbox, setLightbox] = useState<string | null>(null)

  // Load video metadata
  useEffect(() => {
    fetch(`/api/videos/${videoId}`)
      .then(r => r.ok ? r.json() : Promise.reject('Not found'))
      .then(setMeta)
      .catch(() => setError('Video not found'))
  }, [videoId])

  // Try loading cached results
  useEffect(() => {
    fetch(`/api/videos/${videoId}/results`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setResult(data) })
      .catch(() => {})
  }, [videoId])

  const runAnalysis = async () => {
    setAnalyzing(true)
    setError('')
    try {
      const res = await fetch(`/api/videos/${videoId}/analyze`, { method: 'POST' })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || `HTTP ${res.status}`)
      }
      setResult(await res.json())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setAnalyzing(false)
    }
  }

  const formatTime = (sec: number) => {
    const m = Math.floor(sec / 60)
    const s = Math.floor(sec % 60)
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  const handleExport = (type: 'json' | 'csv' | 'zip') => {
    window.open(`/api/export/${videoId}/${type}`, '_blank')
  }

  const handlePrint = () => {
    window.print()
  }

  return (
    <div className="results-page">
      <header className="results-header">
        <a href="#upload" className="back-link">← Upload</a>
        <h1>Analysis Results</h1>
        {meta && <p className="results-subtitle">{meta.original_filename}</p>}
      </header>

      {error && <div className="error-banner">{error}</div>}

      {/* Video info bar */}
      {meta && (
        <div className="video-info-bar">
          <div className="info-chip">
            <span className="chip-label">Resolution</span>
            <span className="chip-value">{meta.width}×{meta.height}</span>
          </div>
          <div className="info-chip">
            <span className="chip-label">Duration</span>
            <span className="chip-value">{formatTime(meta.duration_seconds)}</span>
          </div>
          <div className="info-chip">
            <span className="chip-label">FPS</span>
            <span className="chip-value">{meta.fps}</span>
          </div>
          <div className="info-chip">
            <span className="chip-label">Frames</span>
            <span className="chip-value">{meta.frame_count.toLocaleString()}</span>
          </div>
          <div className="info-chip">
            <span className="chip-label">Status</span>
            <span className={`chip-value status-${result ? 'analyzed' : meta.status}`}>
              {result ? '✅ Analyzed' : meta.status}
            </span>
          </div>
        </div>
      )}

      {/* Run analysis button */}
      {!result && !analyzing && (
        <button className="btn btn-analyze" onClick={runAnalysis}>
          🔍 Run Analysis
        </button>
      )}

      {analyzing && (
        <div className="analyzing-card">
          <div className="spinner" />
          <p>Running analysis pipeline…</p>
          <p className="analyzing-hint">Motion detection & evidence generation</p>
        </div>
      )}

      {/* Results summary & Exports */}
      {result && (
        <>
          <div className="export-toolbar no-print">
            <h3>Diagnostic Report Ready</h3>
            <div className="export-actions">
              <button className="btn btn-outline" onClick={() => handleExport('csv')}>
                📄 CSV
              </button>
              <button className="btn btn-outline" onClick={() => handleExport('json')}>
                📜 JSON
              </button>
              <button className="btn btn-outline" onClick={() => handleExport('zip')}>
                🗂️ Evidence ZIP
              </button>
              <button className="btn btn-print" onClick={handlePrint}>
                🖨️ Print PDF
              </button>
            </div>
          </div>

          <div className="summary-grid">
            <div className="summary-card">
              <span className="summary-value">{result.events.length}</span>
              <span className="summary-label">Events Flagged</span>
            </div>
            <div className="summary-card">
              <span className="summary-value">{result.total_frames_analyzed}</span>
              <span className="summary-label">Frames Analyzed</span>
            </div>
            <div className="summary-card">
              <span className="summary-value">{result.processing_time_seconds.toFixed(2)}s</span>
              <span className="summary-label">Processing Time</span>
            </div>
            <div className="summary-card">
              <span className="summary-value mono">{result.model_version}</span>
              <span className="summary-label">Model</span>
            </div>
          </div>

          {/* Timeline Player */}
          {meta && <TimelinePlayer videoId={videoId} duration={meta.duration_seconds} events={result.events} />}


          {/* Event list */}
          {result.events.length === 0 ? (
            <div className="no-events">
              <p className="no-events-icon">✅</p>
              <p>No suspicious events detected</p>
            </div>
          ) : (
            <div className="event-list">
              <h2 className="section-title">Flagged Events</h2>
              {result.events.map((event) => (
                <div
                  key={event.event_id}
                  className={`event-card ${expandedEvent === event.event_id ? 'event-card--expanded' : ''}`}
                >
                  {/* Event header */}
                  <div
                    className="event-header"
                    onClick={() => setExpandedEvent(
                      expandedEvent === event.event_id ? null : event.event_id
                    )}
                  >
                    <div
                      className="severity-badge"
                      style={{ backgroundColor: SEVERITY_COLORS[event.severity] || '#666' }}
                    >
                      {event.severity.toUpperCase()}
                    </div>
                    <div className="event-title">
                      <span className="event-class">
                        {event.predicted_class.replace(/_/g, ' ')}
                      </span>
                      <span className="event-time">
                        {event.start_timestamp.toFixed(1)}s – {event.end_timestamp.toFixed(1)}s
                      </span>
                    </div>
                    <div className="event-confidence">
                      <div className="conf-bar-track">
                        <div
                          className="conf-bar-fill"
                          style={{ width: `${event.confidence_score * 100}%` }}
                        />
                      </div>
                      <span className="conf-pct">{(event.confidence_score * 100).toFixed(0)}%</span>
                    </div>
                    <span className="expand-icon">
                      {expandedEvent === event.event_id ? '▲' : '▼'}
                    </span>
                  </div>

                  {/* Expanded detail */}
                  {expandedEvent === event.event_id && (
                    <div className="event-detail">
                      <p className="event-explanation">{event.explanation_text}</p>

                      <div className="event-meta-row">
                        <span>Frames: {event.start_frame} – {event.end_frame}</span>
                        <span>Model: {event.model_version}</span>
                      </div>

                      {/* Evidence gallery */}
                      {event.annotated_frame_paths.length > 0 && (
                        <div className="evidence-section">
                          <h4 className="evidence-title">📸 Annotated Evidence</h4>
                          <div className="evidence-gallery">
                            {event.annotated_frame_paths.map((path, i) => (
                              <div
                                key={i}
                                className="evidence-thumb"
                                onClick={() => setLightbox(`/api/videos/${videoId}/evidence/${path.split('/').pop()}`)}
                              >
                                <img
                                  src={`/api/videos/${videoId}/evidence/${path.split('/').pop()}`}
                                  alt={`Evidence frame ${i + 1}`}
                                  loading="lazy"
                                />
                                <span className="thumb-label">Frame {i + 1}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {event.evidence_frame_paths.length > 0 && (
                        <div className="evidence-section">
                          <h4 className="evidence-title">🖼️ Raw Frames</h4>
                          <div className="evidence-gallery">
                            {event.evidence_frame_paths.map((path, i) => (
                              <div
                                key={i}
                                className="evidence-thumb"
                                onClick={() => setLightbox(`/api/videos/${videoId}/evidence/${path.split('/').pop()}`)}
                              >
                                <img
                                  src={`/api/videos/${videoId}/evidence/${path.split('/').pop()}`}
                                  alt={`Raw frame ${i + 1}`}
                                  loading="lazy"
                                />
                                <span className="thumb-label">Raw {i + 1}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Lightbox */}
      {lightbox && (
        <div className="lightbox-overlay" onClick={() => setLightbox(null)}>
          <div className="lightbox-content" onClick={e => e.stopPropagation()}>
            <button className="lightbox-close" onClick={() => setLightbox(null)}>✕</button>
            <img src={lightbox} alt="Evidence full view" />
          </div>
        </div>
      )}
    </div>
  )
}
