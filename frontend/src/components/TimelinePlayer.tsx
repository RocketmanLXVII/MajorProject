import { useState, useRef, useEffect } from 'react'
import './TimelinePlayer.css'

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

interface TimelinePlayerProps {
  videoId: string
  duration: number
  events: FlagEvent[]
}

interface FrameAnnotation {
  frame_index: number
  timestamp_seconds: number
  detections: {
    label: string
    confidence: number
    bbox?: { x: number; y: number; w: number; h: number }
    metadata?: any
  }[]
}

const SEVERITY_COLORS: Record<string, string> = {
  low: '#34d399',
  medium: '#fbbf24',
  high: '#f97316',
  critical: '#ef4444',
}

export default function TimelinePlayer({ videoId, duration, events }: TimelinePlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const timelineRef = useRef<HTMLDivElement>(null)
  
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [hoverTime, setHoverTime] = useState<number | null>(null)
  const [hoverPosition, setHoverPosition] = useState<number>(0)
  
  // Wireframes
  const [showWireframes, setShowWireframes] = useState(false)
  const [annotations, setAnnotations] = useState<FrameAnnotation[]>([])
  const canvasRef = useRef<HTMLCanvasElement>(null)
  
  // Filters
  const [activeSeverities, setActiveSeverities] = useState<Set<string>>(
    new Set(['low', 'medium', 'high', 'critical'])
  )

  const toggleSeverity = (sev: string) => {
    setActiveSeverities(prev => {
      const next = new Set(prev)
      if (next.has(sev)) next.delete(sev)
      else next.add(sev)
      return next
    })
  }

  const filteredEvents = events.filter(e => activeSeverities.has(e.severity))

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const onTimeUpdate = () => setCurrentTime(video.currentTime)
    const onPlay = () => setIsPlaying(true)
    const onPause = () => setIsPlaying(false)

    video.addEventListener('timeupdate', onTimeUpdate)
    video.addEventListener('play', onPlay)
    video.addEventListener('pause', onPause)

    return () => {
      video.removeEventListener('timeupdate', onTimeUpdate)
      video.removeEventListener('play', onPlay)
      video.removeEventListener('pause', onPause)
    }
  }, [])

  // Load annotations
  useEffect(() => {
    fetch(`/api/videos/${videoId}/annotations`)
      .then(r => r.ok ? r.json() : [])
      .then(setAnnotations)
      .catch(() => {})
  }, [videoId])

  // Canvas drawing loop
  useEffect(() => {
    if (!showWireframes || annotations.length === 0) return
    
    let reqId: number

    const SKELETON = [
      [0, 1], [0, 2], [1, 3], [2, 4],  // head
      [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // arms
      [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16] // legs
    ]

    const draw = () => {
      const vid = videoRef.current
      const ctx = canvasRef.current?.getContext('2d')
      if (vid && ctx && canvasRef.current) {
        const time = vid.currentTime
        
        // Find closest annotation frame
        const frame = annotations.reduce((prev, curr) => 
          Math.abs(curr.timestamp_seconds - time) < Math.abs(prev.timestamp_seconds - time) ? curr : prev
        )

        // Make sure canvas matches video view exact size
        canvasRef.current.width = vid.clientWidth
        canvasRef.current.height = vid.clientHeight
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)

        if (Math.abs(frame.timestamp_seconds - time) < 0.5) {
          const scaleX = canvasRef.current.width / vid.videoWidth
          const scaleY = canvasRef.current.height / vid.videoHeight

          frame.detections.forEach(det => {
            // Draw boxes
            if (det.bbox) {
              ctx.strokeStyle = det.label === 'cell phone' ? '#ef4444' : '#f97316'
              ctx.lineWidth = 2
              ctx.strokeRect(det.bbox.x * scaleX, det.bbox.y * scaleY, det.bbox.w * scaleX, det.bbox.h * scaleY)
              
              ctx.fillStyle = ctx.strokeStyle
              ctx.font = '12px Inter'
              ctx.fillText(`${det.label} ${(det.confidence * 100).toFixed(0)}%`, det.bbox.x * scaleX, (det.bbox.y * scaleY) - 5)
            }
            
            // Draw skeleton
            if (det.metadata?.landmarks) {
              const pts = det.metadata.landmarks
              
              ctx.strokeStyle = '#34d399'
              ctx.lineWidth = 1.5
              
              // draw lines
              SKELETON.forEach(([a, b]) => {
                const ptA = pts[String(a)]
                const ptB = pts[String(b)]
                if (ptA?.confidence > 0.4 && ptB?.confidence > 0.4) {
                  ctx.beginPath()
                  ctx.moveTo(ptA.x * scaleX, ptA.y * scaleY)
                  ctx.lineTo(ptB.x * scaleX, ptB.y * scaleY)
                  ctx.stroke()
                }
              })
              
              // draw points
              Object.values(pts).forEach((pt: any) => {
                if (pt.confidence > 0.4) {
                  ctx.fillStyle = '#3b82f6'
                  ctx.beginPath()
                  ctx.arc(pt.x * scaleX, pt.y * scaleY, 3, 0, Math.PI * 2)
                  ctx.fill()
                }
              })
            }
          })
        }
      }
      reqId = requestAnimationFrame(draw)
    }
    
    reqId = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(reqId)
  }, [showWireframes, annotations])

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) videoRef.current.pause()
      else videoRef.current.play()
    }
  }

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!timelineRef.current || !videoRef.current) return
    const rect = timelineRef.current.getBoundingClientRect()
    const pos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    const newTime = pos * duration
    videoRef.current.currentTime = newTime
    setCurrentTime(newTime)
  }

  const handleTimelineMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!timelineRef.current) return
    const rect = timelineRef.current.getBoundingClientRect()
    const pos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    setHoverPosition(pos)
    setHoverTime(pos * duration)
  }

  const handleTimelineMouseLeave = () => {
    setHoverTime(null)
  }

  const seekTo = (timestamp: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = timestamp
      setCurrentTime(timestamp)
      videoRef.current.play()
    }
  }

  const formatTime = (sec: number) => {
    const m = Math.floor(sec / 60)
    const s = Math.floor(sec % 60)
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  return (
    <div className="timeline-player">
      {/* Filters */}
      <div className="player-filters">
        <span className="filter-label">Filter Events:</span>
        {Object.keys(SEVERITY_COLORS).map(sev => (
          <button
            key={sev}
            className={`filter-pill ${activeSeverities.has(sev) ? 'active' : ''}`}
            style={{ 
              borderColor: SEVERITY_COLORS[sev],
              background: activeSeverities.has(sev) ? `${SEVERITY_COLORS[sev]}33` : 'transparent',
              color: activeSeverities.has(sev) ? SEVERITY_COLORS[sev] : 'var(--text-secondary)'
            }}
            onClick={() => toggleSeverity(sev)}
          >
            {sev.toUpperCase()}
          </button>
        ))}
        <button 
          className={`filter-pill ${showWireframes ? 'active wireframe-active' : ''}`}
          onClick={() => setShowWireframes(!showWireframes)}
          style={{ marginLeft: 'auto', background: showWireframes ? 'rgba(52, 211, 153, 0.2)' : 'transparent', borderColor: '#34d399', color: '#34d399' }}
        >
          {showWireframes ? '👁 Wireframes ON' : '👁 Wireframes OFF'}
        </button>
      </div>

      {/* Video Container */}
      <div className="video-container" style={{ position: 'relative' }}>
        <video 
          ref={videoRef}
          className="video-element"
          src={`/api/videos/${videoId}/stream`}
          crossOrigin="anonymous"
          preload="metadata"
          onClick={togglePlay}
        />
        {showWireframes && (
          <canvas 
            ref={canvasRef}
            className="video-overlay-canvas"
            style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
          />
        )}
        <div className={`fast-jump-overlay ${isPlaying ? 'playing' : 'paused'}`}>
          {!isPlaying && <button className="big-play-btn" onClick={togglePlay}>▶</button>}
        </div>
      </div>

      {/* Custom Timeline Controls */}
      <div className="player-controls">
        <button className="control-btn play-btn" onClick={togglePlay}>
          {isPlaying ? '⏸' : '▶'}
        </button>
        
        <span className="time-display">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>

        <div 
          className="timeline-track-container" 
          ref={timelineRef}
          onClick={handleTimelineClick}
          onMouseMove={handleTimelineMouseMove}
          onMouseLeave={handleTimelineMouseLeave}
        >
          <div className="timeline-track">
            {/* Playhead Progress */}
            <div 
              className="timeline-progress" 
              style={{ width: `${(currentTime / duration) * 100}%` }}
            />
            
            {/* Event Markers */}
            {filteredEvents.map(event => {
              const startPct = (event.start_timestamp / duration) * 100
              const widthPct = Math.max(0.5, ((event.end_timestamp - event.start_timestamp) / duration) * 100)
              return (
                <div 
                  key={event.event_id}
                  className="timeline-marker"
                  style={{
                    left: `${startPct}%`,
                    width: `${widthPct}%`,
                    backgroundColor: SEVERITY_COLORS[event.severity] || '#fff'
                  }}
                  onClick={(e) => {
                    e.stopPropagation()
                    seekTo(event.start_timestamp)
                  }}
                  title={`${event.predicted_class} (${event.severity})`}
                />
              )
            })}
          </div>

          {/* Hover Tooltip */}
          {hoverTime !== null && (
            <div 
              className="timeline-hover-tooltip"
              style={{ left: `${hoverPosition * 100}%` }}
            >
              {formatTime(hoverTime)}
            </div>
          )}
        </div>
      </div>

      {/* Interactive Event List mapping to timeline */}
      {filteredEvents.length > 0 && (
        <div className="player-event-list">
          {filteredEvents.map(event => (
            <div 
              key={event.event_id} 
              className="player-event-item" 
              onClick={() => seekTo(event.start_timestamp)}
              style={{ borderLeftColor: SEVERITY_COLORS[event.severity] }}
            >
              <div className="event-item-time">{formatTime(event.start_timestamp)}</div>
              <div className="event-item-info">
                <span className="event-item-class">{event.predicted_class.replace(/_/g, ' ')}</span>
                <span className="event-item-conf">({(event.confidence_score * 100).toFixed(0)}%)</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
