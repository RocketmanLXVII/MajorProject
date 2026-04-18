import { useState, useCallback, useRef } from 'react'
import './UploadPage.css'

interface VideoMeta {
  video_id: string
  original_filename: string
  file_size_bytes: number
  duration_seconds: number
  fps: number
  width: number
  height: number
  frame_count: number
  codec: string
  upload_time: string
  status: string
}

type UploadState = 'idle' | 'uploading' | 'success' | 'error'

export default function UploadPage() {
  const [state, setState] = useState<UploadState>('idle')
  const [progress, setProgress] = useState(0)
  const [video, setVideo] = useState<VideoMeta | null>(null)
  const [error, setError] = useState('')
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const ALLOWED = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

  const handleFile = useCallback(async (file: File) => {
    // Client-side validation
    const ext = '.' + file.name.split('.').pop()?.toLowerCase()
    if (!ALLOWED.includes(ext)) {
      setError(`Unsupported file type "${ext}". Allowed: ${ALLOWED.join(', ')}`)
      setState('error')
      return
    }

    setState('uploading')
    setProgress(0)
    setError('')
    setVideo(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const xhr = new XMLHttpRequest()

      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          setProgress(Math.round((e.loaded / e.total) * 100))
        }
      })

      const response = await new Promise<VideoMeta>((resolve, reject) => {
        xhr.onload = () => {
          if (xhr.status === 201) {
            const data = JSON.parse(xhr.responseText)
            resolve(data.video)
          } else {
            const errData = JSON.parse(xhr.responseText)
            reject(new Error(errData.detail || `Upload failed (HTTP ${xhr.status})`))
          }
        }
        xhr.onerror = () => reject(new Error('Network error'))
        xhr.open('POST', '/api/videos/upload')
        xhr.send(formData)
      })

      setVideo(response)
      setState('success')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
      setState('error')
    }
  }, [])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(false)
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0])
    }
  }, [handleFile])

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(true)
  }, [])

  const onDragLeave = useCallback(() => setDragActive(false), [])

  const onBrowse = () => inputRef.current?.click()

  const onFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0])
    }
  }

  const reset = () => {
    setState('idle')
    setProgress(0)
    setVideo(null)
    setError('')
  }

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const formatDuration = (sec: number) => {
    const m = Math.floor(sec / 60)
    const s = Math.floor(sec % 60)
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  return (
    <div className="upload-page">
      <header className="upload-header">
        <a href="/" className="back-link">← Home</a>
        <h1>Upload Video</h1>
        <p className="upload-subtitle">Upload an exam recording for AI analysis</p>
      </header>

      {state === 'idle' && (
        <div
          className={`drop-zone ${dragActive ? 'drop-zone--active' : ''}`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={onBrowse}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ALLOWED.join(',')}
            onChange={onFileSelect}
            hidden
          />
          <div className="drop-icon">📁</div>
          <p className="drop-text">Drag & drop a video file here</p>
          <p className="drop-hint">or click to browse</p>
          <p className="drop-formats">Supported: MP4, AVI, MOV, MKV, WEBM</p>
        </div>
      )}

      {state === 'uploading' && (
        <div className="upload-progress-card">
          <div className="progress-icon">⬆️</div>
          <p className="progress-label">Uploading & processing…</p>
          <div className="progress-bar-track">
            <div
              className="progress-bar-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="progress-pct">{progress}%</p>
        </div>
      )}

      {state === 'error' && (
        <div className="result-card result-card--error">
          <div className="result-icon">❌</div>
          <p className="result-title">Upload Failed</p>
          <p className="result-detail">{error}</p>
          <button className="btn" onClick={reset}>Try Again</button>
        </div>
      )}

      {state === 'success' && video && (
        <div className="result-card result-card--success">
          <div className="result-icon">✅</div>
          <p className="result-title">Video Ready for Analysis</p>

          <div className="meta-grid">
            <div className="meta-item">
              <span className="meta-label">filename</span>
              <span className="meta-value">{video.original_filename}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">video id</span>
              <span className="meta-value mono">{video.video_id}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">resolution</span>
              <span className="meta-value">{video.width}×{video.height}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">duration</span>
              <span className="meta-value">{formatDuration(video.duration_seconds)}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">fps</span>
              <span className="meta-value">{video.fps}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">frames</span>
              <span className="meta-value">{video.frame_count.toLocaleString()}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">codec</span>
              <span className="meta-value mono">{video.codec}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">file size</span>
              <span className="meta-value">{formatBytes(video.file_size_bytes)}</span>
            </div>
          </div>

          {/* Preview first frame */}
          <div className="frame-preview">
            <p className="meta-label">first frame preview</p>
            <img
              src={`/api/videos/${video.video_id}/frame/0`}
              alt="First frame"
              className="preview-img"
            />
          </div>

          <div className="btn-row">
            <a href={`#results/${video.video_id}`} className="btn btn-primary">🔍 Analyze Now</a>
            <button className="btn btn-secondary" onClick={reset}>Upload Another</button>
          </div>
        </div>
      )}
    </div>
  )
}
