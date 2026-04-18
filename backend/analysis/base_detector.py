"""
MulTiCheat — Base Detector Interface

All detection modules must implement this abstract base class.
This ensures a consistent interface for the pipeline to call into.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from backend.analysis.models import Detection


class BaseDetector(ABC):
    """Abstract base class for all detection modules.

    Every detector receives the current frame and (optionally) the previous
    frame, and returns a list of Detection objects.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this detector."""
        ...

    @property
    def version(self) -> str:
        """Version string for reproducibility tracking."""
        return "0.1.0"

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None,
        frame_index: int,
        fps: float,
    ) -> list[Detection]:
        """Run detection on a single frame.

        Args:
            frame: Current BGR frame (H, W, 3).
            prev_frame: Previous BGR frame, or None for the first frame.
            frame_index: Index of the current frame in the video.
            fps: Video frame rate (used for time conversion).

        Returns:
            List of Detection objects found in this frame.
        """
        ...

    def is_available(self) -> bool:
        """Check if this detector is ready to run (e.g. model loaded)."""
        return True

    def warmup(self) -> None:
        """Optional warmup step (e.g. run a dummy inference to load weights)."""
        pass
