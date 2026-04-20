"""Face detection and gaze estimation using OpenCV Haar cascades.

Uses two cascades already bundled with OpenCV — no downloads needed:
  haarcascade_frontalface_default.xml  — face looking at camera
  haarcascade_profileface.xml          — face looking sideways

Logic:
  frontal face detected  → person is looking AT the robot  (is_frontal=True)
  only profile detected  → person is looking AWAY          (is_frontal=False)
  no face detected       → nobody present                  (face_detected=False)

Runs on every camera frame passed in via process(). Designed to be called
from the AttentionClassifier loop at ~10fps — no internal thread needed.
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FRONTAL_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_PROFILE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

# Minimum face size to count (filters out tiny distant faces)
_MIN_FACE_PX = 60


@dataclass
class GazeResult:
    face_detected: bool
    is_frontal: bool      # True = at least one face looking at robot
    face_count: int       # total faces visible
    frontal_count: int    # faces looking at camera


def estimate_gaze(frame_bgr: np.ndarray) -> GazeResult:
    """Process one camera frame and return gaze estimate."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Equalise histogram for better detection in variable lighting
    gray = cv2.equalizeHist(gray)

    frontal = _FRONTAL_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(_MIN_FACE_PX, _MIN_FACE_PX),
    )
    profile_l = _PROFILE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(_MIN_FACE_PX, _MIN_FACE_PX),
    )
    # Flip frame for right-profile detection
    flipped = cv2.flip(gray, 1)
    profile_r = _PROFILE_CASCADE.detectMultiScale(
        flipped,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(_MIN_FACE_PX, _MIN_FACE_PX),
    )

    n_frontal = len(frontal) if isinstance(frontal, np.ndarray) and len(frontal) else 0
    n_profile = (
        (len(profile_l) if isinstance(profile_l, np.ndarray) else 0) +
        (len(profile_r) if isinstance(profile_r, np.ndarray) else 0)
    )
    total = n_frontal + n_profile

    return GazeResult(
        face_detected=total > 0,
        is_frontal=n_frontal > 0,
        face_count=total,
        frontal_count=n_frontal,
    )
