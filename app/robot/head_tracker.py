"""YOLO-based face detector — same approach as reachy_mini_conversation_app
but without the supervision dependency.

Uses YOLOv11n-face-detection from HuggingFace (~6MB model).
Model is downloaded once and cached in ~/.cache/huggingface/.

Returns face center in pixel coordinates for use with ReachyMini.look_at_image().
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_REPO     = "AdamCodd/YOLOv11n-face-detection"
_MODEL_FILE     = "model.pt"
_CONFIDENCE     = 0.35
_MODEL_PATH: Optional[Path] = None   # cached after first download


def _get_model_path() -> Path:
    global _MODEL_PATH
    if _MODEL_PATH is not None:
        return _MODEL_PATH
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=_MODEL_REPO, filename=_MODEL_FILE)
        _MODEL_PATH = Path(path)
        logger.info("[head_tracker] model at %s", _MODEL_PATH)
        return _MODEL_PATH
    except Exception as e:
        raise RuntimeError(f"[head_tracker] failed to download YOLO model: {e}")


class HeadTracker:
    """Detects faces in camera frames using YOLOv11n.

    Usage:
        tracker = HeadTracker()
        u, v = tracker.detect(frame_bgr)   # pixel coords or (None, None)
    """

    def __init__(self) -> None:
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ultralytics import YOLO
            self._model = YOLO(str(_get_model_path()))
        logger.info("[head_tracker] YOLO model loaded")

    def detect(self, frame_bgr: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        """Detect the best (largest + most confident) face.

        Returns:
            (u, v) pixel coordinates of face center, or (None, None) if no face.
        """
        self._ensure_model()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = self._model(frame_bgr, verbose=False)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, None

        confs = boxes.conf.cpu().numpy()
        xyxys = boxes.xyxy.cpu().numpy()

        # Filter by confidence
        mask = confs >= _CONFIDENCE
        if not mask.any():
            return None, None

        confs = confs[mask]
        xyxys = xyxys[mask]

        # Score = 0.7 * confidence + 0.3 * relative_area
        h, w = frame_bgr.shape[:2]
        frame_area = w * h
        areas = (xyxys[:, 2] - xyxys[:, 0]) * (xyxys[:, 3] - xyxys[:, 1])
        rel_areas = areas / frame_area
        max_area = rel_areas.max() if rel_areas.max() > 0 else 1.0
        scores = 0.7 * confs + 0.3 * (rel_areas / max_area)

        best = np.argmax(scores)
        x1, y1, x2, y2 = xyxys[best]
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
        return u, v

    def face_detected(self, frame_bgr: np.ndarray) -> bool:
        u, _ = self.detect(frame_bgr)
        return u is not None
