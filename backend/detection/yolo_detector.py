"""
MulTiCheat Pro — YOLOv12/v11 + SAHI Sliced Object Detector

Integrates Slicing Aided Hyper Inference (SAHI) for detecting back-row students
and tiny objects (cell phones, chits, calculators, books) in 1080p/4K CCTV frames.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False

from backend.analysis.base_detector import BaseDetector
from backend.analysis.models import BoundingBox, Detection
from backend.config import get_settings

logger = logging.getLogger(__name__)


class YOLOSAHIDetector(BaseDetector):
    """SAHI-boosted YOLO Object Detector for small-object & rear-row student detection."""

    TARGET_CLASSES = {
        0: "person",
        67: "cell phone",
        73: "book",
        63: "laptop",
    }

    def __init__(
        self,
        model_version: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        enable_sahi: bool = True,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_ratio: float = 0.20,
    ):
        self.model_version_str = model_version
        self.conf_threshold = conf_threshold
        self.enable_sahi = enable_sahi
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = overlap_ratio

        self.model: Any = None
        self.sahi_model: Any = None
        self._available = False

        if not ULTRALYTICS_AVAILABLE:
            logger.warning("Ultralytics is not available. YOLOSAHIDetector disabled.")
            return

        try:
            self.model = YOLO(model_version)
            self._available = True
            logger.info("Loaded YOLO model %s successfully.", model_version)

            if SAHI_AVAILABLE and self.enable_sahi:
                try:
                    self.sahi_model = AutoDetectionModel.from_pretrained(
                        model_type="ultralytics",
                        model_path=model_version,
                        confidence_threshold=conf_threshold,
                        device="cpu",
                    )
                    logger.info("Loaded SAHI wrapper for %s.", model_version)
                except Exception as exc:
                    logger.warning("SAHI initialization fallback: %s", exc)
        except Exception as exc:
            logger.error("Failed to load YOLO model %s: %s", model_version, exc)

    @property
    def name(self) -> str:
        return f"YOLO+SAHI Detector ({self.model_version_str})"

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

        detections: List[Detection] = []

        # Run SAHI sliced prediction if enabled
        if SAHI_AVAILABLE and self.sahi_model is not None and self.enable_sahi:
            try:
                sahi_result = get_sliced_prediction(
                    image=frame,
                    detection_model=self.sahi_model,
                    slice_height=self.slice_height,
                    slice_width=self.slice_width,
                    overlap_height_ratio=self.overlap_ratio,
                    overlap_width_ratio=self.overlap_ratio,
                    perform_standard_pred=True,
                    verbose=False,
                )

                for object_prediction in sahi_result.object_prediction_list:
                    cls_id = object_prediction.category.id
                    label_name = object_prediction.category.name.lower()

                    if cls_id not in self.TARGET_CLASSES and label_name not in ("person", "cell phone", "book", "phone"):
                        continue

                    conf = float(object_prediction.score.value)
                    bbox_arr = object_prediction.bbox
                    x1, y1, x2, y2 = int(bbox_arr.minx), int(bbox_arr.miny), int(bbox_arr.maxx), int(bbox_arr.maxy)

                    std_label = self.TARGET_CLASSES.get(cls_id, label_name)
                    if std_label == "cell phone":
                        std_label = "phone"

                    det = Detection(
                        label=std_label,
                        confidence=conf,
                        bbox=BoundingBox(x=x1, y=y1, w=max(1, x2 - x1), h=max(1, y2 - y1)),
                        metadata={"source": "SAHI+YOLO"},
                    )
                    detections.append(det)

                if detections:
                    return detections
            except Exception as exc:
                logger.debug("SAHI execution fallback to standard YOLO: %s", exc)

        # Standard YOLO fallback
        try:
            results = self.model.track(
                frame, persist=True, tracker="botsort.yaml", verbose=False, conf=self.conf_threshold, imgsz=1280
            )
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
                    if label == "cell phone":
                        label = "phone"

                    det = Detection(
                        label=label,
                        confidence=conf,
                        bbox=BoundingBox(x=x1, y=y1, w=max(1, x2 - x1), h=max(1, y2 - y1)),
                        track_id=track_id,
                        metadata={"source": "YOLOv8 Standard"},
                    )
                    detections.append(det)
        except Exception as exc:
            logger.error("Standard YOLO detection error: %s", exc)

        return detections
