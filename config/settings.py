"""Application defaults; matching distance uses metres for geodata and pixels otherwise."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_CHECKPOINT = ROOT / "models" / "best_deeplab.pth"
LEVIR_CHANGE_CHECKPOINT = ROOT / "models" / "BIT_LEVIR.onnx"
MAX_UPLOAD_MB = 180
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU = 0.25
DEFAULT_CENTROID_DISTANCE = 3.0
DEFAULT_AREA_CHANGE = 0.35
TILE_SIZE = 512
TILE_OVERLAP = 128
