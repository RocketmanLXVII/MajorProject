"""
MulTiCheat Pro — RTMPose / YOLO-Pose Estimator Wrapper

Extracts body pose keypoints (COCO / RTMPose keypoints) per tracked student.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from backend.analysis.base_detector import BaseDetector
from backend.analysis.models import BoundingBox, Detection

logger = logging.getLogger(__name__)


class PoseEstimator(BaseDetector):
    """Whole-body pose keypoint estimator."""

    def __init__(self, model_version: str = "yolov8m-pose.pt", conf_threshold: float = 0.35):
        self.model_version_str = model_version
        self.conf_threshold = conf_threshold
        self.model: Any = None
        self._available = False

        if not ULTRALYTICS_AVAILABLE:
            return

        try:
            self.model = YOLO(model_version)
            self._available = True
            logger.info("Loaded Pose Estimator model %s", model_version)
        except Exception as exc:
            logger.error("Failed to load Pose model %s: %s", model_version, exc)

    @property
    def name(self) -> str:
        return f"Pose Estimator ({self.model_version_str})"

    @property
    def version(self) -> str:
        return self.model_version_str

    def is_available(self) -> bool:
        return self._available

    def detect(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray],
        frame_index: int,
        fps: float,
    ) -> List[Detection]:
        if not self._available or self.model is None:
            return []

        results = self.model.track(
            frame, persist=True, tracker="botsort.yaml", verbose=False, conf=self.conf_threshold, imgsz=1280
        )

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None or result.keypoints is None or result.keypoints.data is None:
                continue

            kpts_data = result.keypoints.data.cpu().numpy()
            boxes = result.boxes

            for i, kps in enumerate(kpts_data):
                box = boxes[i]
                track_id = int(box.id[0].item()) if box.id is not None else None
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                kpt_meta: Dict[str, Dict[str, float]] = {}
                for idx, (kx, ky, kconf) in enumerate(kps):
                    kpt_meta[str(idx)] = {"x": float(kx), "y": float(ky), "confidence": float(kconf)}

                det = Detection(
                    label="person",
                    confidence=float(box.conf[0].item()),
                    bbox=BoundingBox(x=x1, y=y1, w=max(1, x2 - x1), h=max(1, y2 - y1)),
                    track_id=track_id,
                    metadata={"landmarks": kpt_meta, "source": "PoseEstimator"},
                )
                detections.append(det)

        return detections
