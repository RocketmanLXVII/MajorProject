"""
MulTiCheat — Pose Detector

Wrapper for Ultralytics YOLOv8-pose to track head and body keypoints and flag suspicious physical behaviors.
Using YOLOv8 for pose replaces MediaPipe for better consistency and performance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from backend.analysis.base_detector import BaseDetector
from backend.analysis.models import Detection

logger = logging.getLogger(__name__)

class PoseDetector(BaseDetector):
    """
    Detector tracking human pose using YOLOv8-pose.
    Evaluates COCO keypoints geometry to detect looking around and suspicious hand movements.
    """

    def __init__(self, model_version: str = "yolov8n-pose.pt", conf_threshold: float = 0.5):
        self.model_version_str = model_version
        self.conf_threshold = conf_threshold
        self.model: Any = None
        self._available = False

        if not ULTRALYTICS_AVAILABLE:
            logger.warning("ultralytics is not installed. PoseDetector will be disabled.")
            return

        try:
            self.model = YOLO(model_version)
            self._available = True
            logger.info("Loaded YOLO Pose model %s successfully.", model_version)
        except Exception as e:
            logger.error("Failed to load YOLO Pose model: %s", e)

    @property
    def name(self) -> str:
        return "YOLO Pose Detector"

    @property
    def version(self) -> str:
        return self.model_version_str

    def is_available(self) -> bool:
        return self._available

    def warmup(self) -> None:
        if self._available and self.model is not None:
            dummy = np.zeros((360, 640, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)
            logger.debug("%s warmed up.", self.name)

    def detect(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None,
        frame_index: int,
        fps: float,
    ) -> list[Detection]:
        if not self._available or self.model is None:
            return []

        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=self.conf_threshold)
        
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None or result.keypoints is None or result.keypoints.xy is None:
                continue

            kpts_data = result.keypoints.data.cpu().numpy()  # (N, 17, 3) 
            boxes_data = result.boxes

            for i, person_kps in enumerate(kpts_data):
                box = boxes_data[i]
                
                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0].item())

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                bbox = BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

                kpt_meta = {}
                for idx, (x, y, conf) in enumerate(person_kps):
                    kpt_meta[str(idx)] = {"x": float(x), "y": float(y), "confidence": float(conf)}
                
                # Yield the person boundary with keypoints
                detections.append(Detection(
                    label="person",
                    confidence=float(box.conf[0].item()),
                    bbox=bbox,
                    track_id=track_id,
                    metadata={"landmarks": kpt_meta, "source": "YOLO-Pose"}
                ))
                
                # Heuristics checking using COCO indices
                def get_kp(idx: int) -> tuple[float, float, float]:
                    kp = kpt_meta.get(str(idx), {})
                    return kp.get("x", 0.0), kp.get("y", 0.0), kp.get("confidence", 0.0)
                    
                nose_x, nose_y, nose_c = get_kp(0)
                ls_x, ls_y, ls_c = get_kp(5)
                rs_x, rs_y, rs_c = get_kp(6)
                lw_x, lw_y, lw_c = get_kp(9)
                rw_x, rw_y, rw_c = get_kp(10)
                
                # LOOK_AROUND logic (turning head)
                if nose_c > 0.4 and ls_c > 0.4 and rs_c > 0.4:
                    shoulder_width = abs(ls_x - rs_x)
                    shoulder_center = (ls_x + rs_x) / 2.0
                    if shoulder_width > 10:
                        offset_ratio = abs(nose_x - shoulder_center) / shoulder_width
                        if offset_ratio > 0.45:
                            detections.append(Detection(
                                label="look_around",
                                confidence=min(1.0, offset_ratio * 1.5),
                                track_id=track_id,
                                bbox=bbox,
                                metadata={"offset_ratio": offset_ratio}
                            ))
                
                # SUSPICIOUS_HAND_MOVEMENT logic
                hand_conf = 0.0
                if ls_c > 0.4 and rs_c > 0.4:
                    shoulder_width = abs(ls_x - rs_x)
                    shoulder_center = (ls_x + rs_x) / 2.0
                    
                    if lw_c > 0.4:
                        if (shoulder_center - lw_x) / (shoulder_width + 1) > 1.4:
                            hand_conf = max(hand_conf, 0.6)
                    if rw_c > 0.4:
                        if (rw_x - shoulder_center) / (shoulder_width + 1) > 1.4:
                            hand_conf = max(hand_conf, 0.6)
                            
                if hand_conf > 0:
                    detections.append(Detection(
                        label="suspicious_hand_movement",
                        confidence=hand_conf,
                        track_id=track_id,
                        bbox=bbox,
                    ))

        return detections
