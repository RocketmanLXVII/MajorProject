"""
MulTiCheat Pro — PyTorch Temporal LSTM Pose Classifier

Classifies temporal 3D pose sequences (30-60 frame sliding window) into candidate
malpractice behavior states using a lightweight PyTorch LSTM architecture.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class PoseLSTMPyTorch(nn.Module):
        """Lightweight 2-layer LSTM for 3D Pose Sequence Classification."""

        def __init__(self, input_dim: int = 34, hidden_dim: int = 64, num_classes: int = 10):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: [batch, sequence_length, input_dim]
            lstm_out, _ = self.lstm(x)
            # Use last time-step feature vector
            last_out = lstm_out[:, -1, :]
            logits = self.fc(last_out)
            return logits


class TemporalPoseClassifier:
    """Classifies sequence of 2D/3D pose keypoints into behavior probabilities."""

    CLASSES = [
        "NORMAL",
        "PEEKING_NEIGHBOR",
        "PHONE_USAGE",
        "PAPER_COPYING",
        "CHIT_USAGE",
        "NOTE_PASSING",
        "TURNING_AROUND",
        "TALKING_SIGNALING",
        "STAND_LEAN",
        "UNAUTHORIZED_MATERIAL",
    ]

    def __init__(self, model_path: Optional[str] = None, window_size: int = 30):
        self.window_size = window_size
        self.model = None
        self.device = "cpu"

        if HAS_TORCH and model_path and os.path.exists(model_path):
            try:
                self.model = PoseLSTMPyTorch(input_dim=34, hidden_dim=64, num_classes=len(self.CLASSES))
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            except Exception as e:
                print(f"[TemporalPoseClassifier] Could not load model from {model_path}: {e}")
                self.model = None

    def classify_sequence(
        self,
        pose_sequence: List[List[float]],  # Shape: [num_frames, 34] (e.g. 17 keypoint (x,y) pairs)
    ) -> Dict[str, float]:
        """Predict behavior probabilities for a sequence of keypoint vectors."""
        if not pose_sequence or len(pose_sequence) == 0:
            return {cls_name: 0.1 for cls_name in self.CLASSES}

        # Padding or truncating sequence to self.window_size
        seq = np.array(pose_sequence, dtype=np.float32)
        if seq.shape[0] < self.window_size:
            pad = np.repeat(seq[-1:], self.window_size - seq.shape[0], axis=0)
            seq = np.vstack([seq, pad])
        else:
            seq = seq[-self.window_size:]

        if HAS_TORCH and self.model is not None:
            with torch.no_grad():
                tensor_in = torch.from_numpy(seq).unsqueeze(0).to(self.device)  # [1, T, 34]
                logits = self.model(tensor_in)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                return {cls_name: float(probs[i]) for i, cls_name in enumerate(self.CLASSES)}

        # Fallback heuristic prediction if weights not present
        return self._fallback_heuristic(seq)

    def _fallback_heuristic(self, seq: np.ndarray) -> Dict[str, float]:
        """Simple statistical feature classifier fallback."""
        # Mean variance across keypoint motion
        var_motion = np.var(seq, axis=0).mean()
        probs = {cls_name: 0.05 for cls_name in self.CLASSES}
        probs["NORMAL"] = 0.70

        if var_motion > 50.0:
            probs["TURNING_AROUND"] = 0.60
            probs["NORMAL"] = 0.20
        elif var_motion > 25.0:
            probs["PEEKING_NEIGHBOR"] = 0.50
            probs["NORMAL"] = 0.30

        return probs
