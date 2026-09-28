"""
MulTiCheat — Configuration & Profile System

Supports profiles (FAST, BALANCED, ACCURACY, BENCHMARK) and loads settings from
config.yaml / environment variables (MULTICHEAT_*).
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ModelProfile(str, Enum):
    FAST = "FAST"
    BALANCED = "BALANCED"
    ACCURACY = "ACCURACY"
    BENCHMARK = "BENCHMARK"


def load_yaml_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        logger.warning("config.yaml not found at %s — using defaults.", config_path)
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error("Failed to parse config.yaml: %s", exc)
        return {}


_yaml = load_yaml_config()


class PerceptionConfig:
    """Configurable perception parameters."""

    def __init__(
        self,
        profile: ModelProfile = ModelProfile.BALANCED,
        detector_model: str = "yolov8n.pt",
        pose_model: str = "yolov8m-pose.pt",
        segmentation_model: str = "yolov8n-seg.pt",
        full_frame_imgsz: int = 1280,
        tile_imgsz: int = 1280,
        tile_rows: int = 2,
        tile_cols: int = 2,
        tile_overlap: float = 0.20,
        enable_tiling: bool = True,
        enable_student_crops: bool = True,
        crop_expand_ratio: float = 0.25,
        crop_min_size: int = 224,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ):
        self.profile = profile
        self.detector_model = detector_model
        self.pose_model = pose_model
        self.segmentation_model = segmentation_model
        self.full_frame_imgsz = full_frame_imgsz
        self.tile_imgsz = tile_imgsz
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.tile_overlap = tile_overlap
        self.enable_tiling = enable_tiling
        self.enable_student_crops = enable_student_crops
        self.crop_expand_ratio = crop_expand_ratio
        self.crop_min_size = crop_min_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    @classmethod
    def from_profile(cls, profile: ModelProfile) -> PerceptionConfig:
        if profile == ModelProfile.FAST:
            return cls(
                profile=profile,
                detector_model="yolov8n.pt",
                pose_model="yolov8n-pose.pt",
                full_frame_imgsz=640,
                enable_tiling=False,
                enable_student_crops=True,
                conf_threshold=0.30,
            )
        elif profile == ModelProfile.ACCURACY:
            return cls(
                profile=profile,
                detector_model="yolov8m.pt",
                pose_model="yolov8m-pose.pt",
                full_frame_imgsz=1280,
                tile_rows=3,
                tile_cols=3,
                tile_overlap=0.25,
                enable_tiling=True,
                enable_student_crops=True,
                conf_threshold=0.20,
            )
        elif profile == ModelProfile.BENCHMARK:
            return cls(
                profile=profile,
                detector_model="yolov8m.pt",
                pose_model="yolov8m-pose.pt",
                full_frame_imgsz=1280,
                tile_rows=3,
                tile_cols=3,
                tile_overlap=0.30,
                enable_tiling=True,
                enable_student_crops=True,
                conf_threshold=0.15,
            )
        else:  # BALANCED
            return cls(
                profile=profile,
                detector_model="yolov8n.pt",
                pose_model="yolov8m-pose.pt",
                full_frame_imgsz=1280,
                tile_rows=2,
                tile_cols=2,
                tile_overlap=0.20,
                enable_tiling=True,
                enable_student_crops=True,
                conf_threshold=0.25,
            )


class Settings(BaseSettings):
    """Central Application Settings."""

    model_config = SettingsConfigDict(
        env_prefix="MULTICHEAT_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default=_yaml.get("app", {}).get("name", "MulTiCheat"))
    app_version: str = Field(default=_yaml.get("app", {}).get("version", "0.2.0"))
    debug: bool = Field(default=_yaml.get("app", {}).get("debug", False))

    host: str = Field(default=_yaml.get("server", {}).get("host", "0.0.0.0"))
    port: int = Field(default=_yaml.get("server", {}).get("port", 8000))

    upload_dir: str = Field(default=_yaml.get("storage", {}).get("upload_dir", "uploads"))
    evidence_dir: str = Field(default=_yaml.get("storage", {}).get("evidence_dir", "evidence"))
    db_path: str = Field(default=_yaml.get("storage", {}).get("db_path", "multicheat.db"))
    max_upload_size_mb: int = Field(default=_yaml.get("storage", {}).get("max_upload_size_mb", 500))

    active_profile: ModelProfile = Field(default=ModelProfile.BALANCED)
    log_level: str = Field(default=_yaml.get("logging", {}).get("level", "INFO"))
    log_format: str = Field(
        default=_yaml.get(
            "logging",
            {},
        ).get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def evidence_path(self) -> Path:
        p = Path(self.evidence_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def db_file_path(self) -> Path:
        p = Path(self.db_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def ensure_directories(self) -> None:
        self.upload_path.mkdir(parents=True, exist_ok=True)
        self.evidence_path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    if not hasattr(get_settings, "_instance"):
        get_settings._instance = Settings()  # type: ignore
    return get_settings._instance  # type: ignore
