"""
MulTiCheat Pro — 3D Geometry & Ray-Plane Intersection Utilities

Implements:
- 5.1 Gaze Ray to Desk Polygon Intersection
- 5.2 Perspective Pixel-to-World Coordinate Conversion
- 5.3 Calibrated Behavior Confidence Scoring
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


def pixel_to_world(
    pixel_point: Tuple[float, float],
    homography_matrix: np.ndarray,
    camera_height_m: float = 2.8,
) -> np.ndarray:
    """
    Convert image pixel (x, y) to world ground-plane coordinate (x, z) in meters
    using inverse homography.
    """
    if homography_matrix is None or homography_matrix.shape != (3, 3):
        # Default identity mapping with unit scale factor
        return np.array([pixel_point[0] / 100.0, pixel_point[1] / 100.0])

    try:
        H_inv = np.linalg.inv(homography_matrix)
        pt = np.array([pixel_point[0], pixel_point[1], 1.0], dtype=np.float32)
        world_h = H_inv @ pt
        if abs(world_h[2]) > 1e-6:
            world_x = world_h[0] / world_h[2]
            world_z = world_h[1] / world_h[2]
            return np.array([float(world_x), float(world_z)])
    except Exception:
        pass

    return np.array([pixel_point[0] / 100.0, pixel_point[1] / 100.0])


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Check if 2D point (x, y) is inside a 2D polygon using OpenCV pointPolygonTest."""
    poly_arr = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
    dist = cv2.pointPolygonTest(poly_arr, (float(point[0]), float(point[1])), False)
    return dist >= 0


def intersect_gaze_with_desks(
    eye_center_3d: np.ndarray,   # [x, y, z] in meters
    gaze_vector: np.ndarray,     # [dx, dy, dz] unit vector
    desks: List[Dict[str, Any]],
    max_distance: float = 3.0,
    desk_height_m: float = 0.75,
) -> Optional[Dict[str, Any]]:
    """
    Section 5.1: Cast ray P = eye_center + t * gaze_vector onto desk height plane y = desk_height.
    Return closest intersected desk within max_distance.
    """
    if gaze_vector is None or len(gaze_vector) < 3:
        return None

    eye_x, eye_y, eye_z = eye_center_3d[0], eye_center_3d[1], eye_center_3d[2]
    gdx, gdy, gdz = gaze_vector[0], gaze_vector[1], gaze_vector[2]

    # If gaze_dy >= 0 (looking upwards or horizontal), no desk plane intersection
    if gdy >= -0.05:
        return None

    # Solve for t: eye_y + t * gdy = desk_height_m
    t = (desk_height_m - eye_y) / gdy
    if t <= 0 or t > max_distance:
        return None

    # Intersection point P = eye_center + t * gaze_vector
    intersect_pt = np.array([
        eye_x + t * gdx,
        desk_height_m,
        eye_z + t * gdz,
    ])

    closest_desk = None
    min_dist = float("inf")

    for desk in desks:
        bbox = desk.get("bbox")  # [x, y, w, h] or polygon coordinates
        poly = desk.get("polygon")
        if not poly and bbox:
            x, y, w, h = bbox
            poly = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

        if poly:
            if point_in_polygon((intersect_pt[0], intersect_pt[2]), poly):
                if t < min_dist:
                    min_dist = t
                    closest_desk = desk

    return closest_desk


def compute_behavior_confidence(
    detector_confidence: float,
    temporal_consistency: float,
    geometric_plausibility: float,
    calibration_quality: float = 1.0,
    w1: float = 0.30,
    w2: float = 0.30,
    w3: float = 0.25,
    w4: float = 0.15,
) -> float:
    """
    Section 5.3: Calibrated confidence formula:
    confidence = w1*det + w2*temp + w3*geom + w4*calib
    """
    score = (
        (w1 * min(1.0, max(0.0, detector_confidence)))
        + (w2 * min(1.0, max(0.0, temporal_consistency)))
        + (w3 * min(1.0, max(0.0, geometric_plausibility)))
        + (w4 * min(1.0, max(0.0, calibration_quality)))
    )
    return round(min(1.0, max(0.0, score)), 3)
