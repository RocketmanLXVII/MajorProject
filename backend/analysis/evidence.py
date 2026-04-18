"""
MulTiCheat — Evidence Frame Generation (Phase 3 Enhanced)

Rich visual annotations with:
- Color-coded bounding boxes by severity/label
- Confidence score bars beside each detection
- Semi-transparent severity banner at the top
- Event summary panel at the bottom
- Keypoint visualization support (for future pose detection)
- Frame timestamp and score gauge
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from backend.analysis.models import Detection, FlagCategory, FlagEvent, Severity
from backend.config import get_settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Color palette (BGR for OpenCV)
# ──────────────────────────────────────────────

SEVERITY_COLORS = {
    Severity.LOW:      (180, 180, 60),    # teal-ish
    Severity.MEDIUM:   (0, 200, 255),     # orange
    Severity.HIGH:     (0, 100, 255),     # red-orange
    Severity.CRITICAL: (0, 0, 255),       # red
}

LABEL_COLORS = {
    "motion":              (200, 180, 0),    # cyan-ish
    "large_motion":        (0, 140, 255),    # orange
    "high_overall_motion": (50, 50, 255),    # red
    "person":              (255, 160, 0),    # blue
    "phone":               (0, 0, 255),      # red
    "book":                (0, 200, 255),    # yellow
    "laptop":              (0, 200, 255),    # yellow
    "default":             (200, 160, 40),   # blue-ish
}

KEYPOINT_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0),
]


def _get_label_color(label: str) -> tuple[int, int, int]:
    return LABEL_COLORS.get(label, LABEL_COLORS["default"])


def _get_severity_color(severity: Severity) -> tuple[int, int, int]:
    return SEVERITY_COLORS.get(severity, SEVERITY_COLORS[Severity.LOW])


# ──────────────────────────────────────────────
# Core annotation
# ──────────────────────────────────────────────

def annotate_frame(
    frame: np.ndarray,
    detections: list[Detection],
    frame_index: int = 0,
    timestamp: float = 0.0,
    aggregate_score: float = 0.0,
    event: Optional[FlagEvent] = None,
) -> np.ndarray:
    """
    Draw rich annotations on a frame.

    Includes bounding boxes, labels, confidence bars,
    severity banner, and score gauge.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # ── Draw detections ────────────────────
    for det in detections:
        color = _get_label_color(det.label)
        _draw_detection(annotated, det, color)

    # ── Draw keypoints if present ──────────
    for det in detections:
        if "keypoints" in det.metadata:
            _draw_keypoints(annotated, det.metadata["keypoints"])

    # ── Top info bar ───────────────────────
    _draw_info_bar(annotated, frame_index, timestamp, aggregate_score, event)

    # ── Score gauge (bottom-right) ─────────
    _draw_score_gauge(annotated, aggregate_score)

    # ── Event summary panel (if event) ─────
    if event is not None:
        _draw_event_panel(annotated, event)

    return annotated


def annotate_frame_simple(
    frame: np.ndarray,
    detections: list[Detection],
    frame_index: int = 0,
    timestamp: float = 0.0,
) -> np.ndarray:
    """Simplified annotation — boxes and labels only. Fast path for bulk frames."""
    annotated = frame.copy()
    for det in detections:
        color = _get_label_color(det.label)
        if det.bbox is not None:
            x, y, bw, bh = det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
            label_text = f"{det.label} {det.confidence:.0%}"
            cv2.putText(annotated, label_text, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return annotated


# ──────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────

def _draw_detection(img: np.ndarray, det: Detection, color: tuple) -> None:
    """Draw a bounding box with label and confidence bar."""
    if det.bbox is None:
        return

    x, y, bw, bh = det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h
    h_img, w_img = img.shape[:2]

    # Clamp to image bounds
    x2 = min(x + bw, w_img)
    y2 = min(y + bh, h_img)

    # Box with rounded corners effect (double-line)
    cv2.rectangle(img, (x, y), (x2, y2), color, 2)
    # Inner glow line
    cv2.rectangle(img, (x + 1, y + 1), (x2 - 1, y2 - 1),
                  tuple(min(255, c + 40) for c in color), 1)

    # Label pill
    label_text = f"{det.label} {det.confidence:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label_text, font, scale, thickness)

    pill_y = max(y - th - 10, 0)
    cv2.rectangle(img, (x, pill_y), (x + tw + 8, pill_y + th + 8), color, -1)
    cv2.putText(img, label_text, (x + 4, pill_y + th + 4),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Confidence bar (right side of box)
    bar_x = x2 + 4
    bar_h = min(bh, 60)
    bar_w = 6
    if bar_x + bar_w < w_img:
        fill_h = int(bar_h * det.confidence)
        cv2.rectangle(img, (bar_x, y + bar_h - fill_h), (bar_x + bar_w, y + bar_h),
                      color, -1)
        cv2.rectangle(img, (bar_x, y), (bar_x + bar_w, y + bar_h),
                      (100, 100, 100), 1)


def _draw_keypoints(
    img: np.ndarray,
    keypoints: dict[str, dict],
) -> None:
    """Draw keypoints and skeleton connections.

    Each keypoint should be tracked by index string ('0', '1', etc) with 'x', 'y' and 'confidence'.
    """
    pts = {}
    for i_str, kp in keypoints.items():
        if not isinstance(kp, dict) or "x" not in kp or "y" not in kp:
            continue
            
        i = int(i_str)
        kx, ky = int(kp["x"]), int(kp["y"])
        conf = kp.get("confidence", 1.0)
        
        if conf < 0.3:
            continue
            
        color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
        radius = 4 if conf > 0.6 else 2
        cv2.circle(img, (kx, ky), radius, color, -1)
        pts[i] = (kx, ky)

    # Draw skeleton lines between adjacent keypoints
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    ]
    for a, b in SKELETON:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], (180, 180, 180), 1, cv2.LINE_AA)


def _draw_info_bar(
    img: np.ndarray,
    frame_index: int,
    timestamp: float,
    score: float,
    event: Optional[FlagEvent],
) -> None:
    """Draw a semi-transparent info bar at the top of the frame."""
    h, w = img.shape[:2]
    bar_h = 32

    # Semi-transparent overlay
    overlay = img.copy()
    bar_color = (30, 30, 50)
    if event is not None:
        sev_color = _get_severity_color(event.severity)
        bar_color = tuple(c // 3 for c in sev_color)
    cv2.rectangle(overlay, (0, 0), (w, bar_h), bar_color, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    info = f"Frame {frame_index}  |  {timestamp:.2f}s  |  Score: {score:.0%}"
    cv2.putText(img, info, (10, 22), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    # Severity badge (right side)
    if event is not None:
        sev_text = f"  {event.severity.value.upper()}  "
        sev_color = _get_severity_color(event.severity)
        (stw, sth), _ = cv2.getTextSize(sev_text, font, 0.5, 1)
        badge_x = w - stw - 14
        cv2.rectangle(img, (badge_x, 4), (badge_x + stw + 8, 28), sev_color, -1)
        cv2.putText(img, sev_text, (badge_x + 4, 22), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)


def _draw_score_gauge(img: np.ndarray, score: float) -> None:
    """Draw a circular score gauge in the bottom-right corner."""
    h, w = img.shape[:2]
    center = (w - 40, h - 40)
    radius = 25

    # Background circle
    cv2.circle(img, center, radius, (40, 40, 40), -1)
    cv2.circle(img, center, radius, (80, 80, 80), 1)

    # Arc proportional to score
    angle = int(360 * score)
    if score < 0.3:
        arc_color = (0, 200, 0)     # green
    elif score < 0.6:
        arc_color = (0, 200, 255)   # orange
    else:
        arc_color = (0, 0, 255)     # red

    cv2.ellipse(img, center, (radius - 3, radius - 3),
                -90, 0, angle, arc_color, 3, cv2.LINE_AA)

    # Score text
    cv2.putText(img, f"{score:.0%}", (center[0] - 16, center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_event_panel(img: np.ndarray, event: FlagEvent) -> None:
    """Draw an event summary panel at the bottom of the frame."""
    h, w = img.shape[:2]
    panel_h = 48
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Semi-transparent panel
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    # Event info
    class_text = event.predicted_class.value.replace("_", " ").title()
    line1 = f"Event: {class_text}  |  Conf: {event.confidence_score:.0%}  |  {event.start_timestamp:.1f}s - {event.end_timestamp:.1f}s"
    cv2.putText(img, line1, (10, h - panel_h + 18), font, 0.42,
                (220, 220, 220), 1, cv2.LINE_AA)

    # Severity color bar
    sev_color = _get_severity_color(event.severity)
    cv2.rectangle(img, (0, h - 4), (w, h), sev_color, -1)


# ──────────────────────────────────────────────
# Evidence directory and saving
# ──────────────────────────────────────────────

def get_evidence_dir(video_id: str) -> Path:
    """Get or create the evidence directory for a video."""
    settings = get_settings()
    evidence_dir = settings.evidence_path / video_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    return evidence_dir


def save_evidence_frame(
    frame: np.ndarray,
    video_id: str,
    event_id: str,
    frame_index: int,
    annotated: bool = True,
) -> str:
    """
    Save a single evidence frame to disk as JPEG.

    Returns the relative path from the evidence root (e.g. "vid123/annotated_evt001_f000042.jpg").
    """
    evidence_dir = get_evidence_dir(video_id)
    prefix = "annotated" if annotated else "raw"
    filename = f"{prefix}_{event_id}_f{frame_index:06d}.jpg"
    filepath = evidence_dir / filename

    success = cv2.imwrite(
        str(filepath), frame,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )

    if not success:
        logger.error("Failed to save evidence frame: %s", filepath)
        return ""

    rel_path = f"{video_id}/{filename}"
    logger.debug("Saved evidence frame: %s", rel_path)
    return rel_path


def save_event_evidence(
    frames: list[tuple[int, float, np.ndarray, list[Detection]]],
    video_id: str,
    event_id: str,
    event: Optional[FlagEvent] = None,
    aggregate_scores: Optional[dict[int, float]] = None,
) -> tuple[list[str], list[str]]:
    """
    Save evidence frames for a single event.

    Args:
        frames: list of (frame_index, timestamp, frame, detections)
        event: optional FlagEvent for rich annotation context
        aggregate_scores: optional map of frame_index → score

    Returns:
        (raw_evidence_paths, annotated_evidence_paths)
    """
    evidence_paths: list[str] = []
    annotated_paths: list[str] = []
    scores = aggregate_scores or {}

    for frame_index, timestamp, frame, detections in frames:
        # Save raw evidence
        raw_path = save_evidence_frame(
            frame, video_id, event_id, frame_index, annotated=False
        )
        if raw_path:
            evidence_paths.append(raw_path)

        # Save annotated evidence (rich version)
        score = scores.get(frame_index, 0.0)
        annotated_frame = annotate_frame(
            frame, detections,
            frame_index=frame_index,
            timestamp=timestamp,
            aggregate_score=score,
            event=event,
        )
        ann_path = save_evidence_frame(
            annotated_frame, video_id, event_id, frame_index, annotated=True
        )
        if ann_path:
            annotated_paths.append(ann_path)

    return evidence_paths, annotated_paths


# ──────────────────────────────────────────────
# Evidence gallery listing
# ──────────────────────────────────────────────

def list_evidence_files(video_id: str) -> list[dict]:
    """List all evidence files for a video with metadata."""
    settings = get_settings()
    evidence_dir = settings.evidence_path / video_id

    if not evidence_dir.exists():
        return []

    files = []
    for fp in sorted(evidence_dir.iterdir()):
        if fp.is_file() and fp.suffix.lower() in (".jpg", ".jpeg", ".png"):
            # Parse filename: {type}_{event_id}_f{frame_idx}.jpg
            name = fp.stem
            parts = name.split("_", 2)
            file_info = {
                "filename": fp.name,
                "type": parts[0] if parts else "unknown",
                "event_id": parts[1] if len(parts) > 1 else "unknown",
                "size_bytes": fp.stat().st_size,
                "path": f"{video_id}/{fp.name}",
            }
            # Extract frame index
            if len(parts) > 2 and parts[2].startswith("f"):
                try:
                    file_info["frame_index"] = int(parts[2][1:])
                except ValueError:
                    pass
            files.append(file_info)

    return files
