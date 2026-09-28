"""
MulTiCheat Pro — Model Pre-fetching & Download Utility

Pre-downloads required YOLOv8 and MediaPipe model files into local models/ directory.
"""

import os
import sys
import urllib.request
from pathlib import Path


MODEL_DIR = Path(__file__).parent / "models"

MODELS_TO_DOWNLOAD = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt",
}


def download_models():
    """Download required weights if missing."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Checking model directory: {MODEL_DIR.resolve()}")
    for filename, url in MODELS_TO_DOWNLOAD.items():
        dest = MODEL_DIR / filename
        if dest.exists() and dest.stat().st_size > 100000:
            print(f"[EXISTS] {filename} ({dest.stat().st_size / (1024*1024):.1f} MB)")
        else:
            print(f"[DOWNLOADING] {filename} from {url}...")
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"[DOWNLOADED] {filename} successfully!")
            except Exception as e:
                print(f"[ERROR] Failed downloading {filename}: {e}")

    print("\nModel pre-download complete!")


if __name__ == "__main__":
    download_models()
