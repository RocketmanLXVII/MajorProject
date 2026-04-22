"""
MulTiCheat — YOLO Object Detector

Wrapper for Ultralytics YOLOv8 for detecting unauthorized objects (e.g., cell phones, notes/books).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

# Lazy load ultralytics to avoid heavy import overhead if unused
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from backend.analysis.base_detector import BaseDetector
from backend.analysis.models import BoundingBox, Detection

logger = logging.getLogger(__name__)

class YOLODetector(BaseDetector):
    """
    Object detector using YOLOv8 to find specific objects.
    Target objects for cheating detection: 'cell phone', 'book'.
    """

    # COCO dataset standard class mapping
    TARGET_CLASSES = {
        67: "cell phone",
        73: "book"
    }

    def __init__(self, model_version: str = "yolov8m.pt", conf_threshold: float = 0.3):
        """
        Initialize the YOLO detector.
        
        Args:
            model_version: The weights to load (e.g., 'yolov8n.pt' for nano, 'yolov8s.pt' for small).
                           Will automatically download if not present.
            conf_threshold: Minimum confidence bound for reporting a detection.
        """
        self.model_version_str = model_version
        self.conf_threshold = conf_threshold
        self.model: Any = None
        self._available = False

        if not ULTRALYTICS_AVAILABLE:
            logger.warning("ultralytics is not installed. YOLODetector will be disabled.")
            return

        try:
            self.model = YOLO(model_version)
            self._available = True
            logger.info("Loaded YOLO model %s successfully.", model_version)
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)

    @property
    def name(self) -> str:
        return "YOLO Object Detector"

    @property
    def version(self) -> str:
        return self.model_version_str

    def is_available(self) -> bool:
        return self._available

    def warmup(self) -> None:
        if self._available and self.model is not None:
            # Run inference on a dummy array to initialize the network graph and CUDA (if present)
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
        """
        Run object detection on the current frame.
        Filter for target classes (cell phone, book).
        """
        if not self._available or self.model is None:
            return []

        # Run inference specifying classes of interest using BoT-SORT at high-resolution
        results = self.model.track(frame, persist=True, tracker="botsort.yaml", verbose=False, conf=self.conf_threshold, imgsz=1280)
        
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in self.TARGET_CLASSES:
                    continue
                
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0].item())

                label = self.TARGET_CLASSES[cls_id]

                det = Detection(
                    label=label,
                    confidence=conf,
                    bbox=BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                    track_id=track_id,
                    metadata={"source": "YOLOv8 Object Tracker"}
                )
                detections.append(det)

        return detections
