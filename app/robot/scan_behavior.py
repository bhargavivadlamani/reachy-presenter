"""Scanning behavior — robot looks around until it sees a face.

Uses the same approach as reachy_mini_conversation_app:
  - YOLO face detection (HeadTracker)
  - ReachyMini.look_at_image() for precise head aiming

Scan pattern:
  center → left → center → right → center → repeat

When face is detected:
  - Stop scanning
  - Call look_at_image(u, v) to aim at the face
  - Fire on_face_found() callback

When face disappears for GONE_SECONDS:
  - Resume scanning
  - Fire on_face_lost() callback
"""

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

from app.robot.head_tracker import HeadTracker

logger = logging.getLogger(__name__)

# Scan head positions as (yaw_deg, pitch_deg)
_SCAN_POSES = [
    (0,   -8),    # center slightly down (faces are ahead)
    (40,  -8),    # left
    (0,   -8),    # center
    (-40, -8),    # right
    (0,   -8),    # center
]
_SCAN_STEP_SEC  = 1.5    # seconds per scan step
_GONE_SECONDS   = 8.0    # seconds without face before resuming scan
_POLL_INTERVAL  = 0.12   # seconds between camera checks


class ScanBehavior:
    """Physically scans the environment with head + YOLO face detection.

    When a face is found, aims the head at it using look_at_image().
    """

    def __init__(
        self,
        mini,
        on_face_found: Optional[Callable[[], None]] = None,
        on_face_lost:  Optional[Callable[[], None]] = None,
    ) -> None:
        self._mini = mini
        self._on_face_found = on_face_found
        self._on_face_lost = on_face_lost
        self._tracker = HeadTracker()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.face_visible = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("[scan] behavior started")

    def stop(self) -> None:
        self._stop.set()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    # ------------------------------------------------------------------

    def _run(self) -> None:
        face_lost_at: Optional[float] = None

        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.2)
                continue

            frame = self._mini.media.get_frame()
            if frame is None:
                time.sleep(_POLL_INTERVAL)
                continue

            u, v = self._tracker.detect(frame)
            now = time.time()

            if u is not None:
                face_lost_at = None
                if not self.face_visible:
                    self.face_visible = True
                    logger.info("[scan] face detected at (%d, %d)", u, v)
                    self._aim_at_face(u, v)
                    if self._on_face_found:
                        threading.Thread(
                            target=self._on_face_found, daemon=True
                        ).start()
                else:
                    # Keep tracking — smoothly follow face
                    self._aim_at_face(u, v)
            else:
                if self.face_visible:
                    if face_lost_at is None:
                        face_lost_at = now
                    elif now - face_lost_at >= _GONE_SECONDS:
                        self.face_visible = False
                        face_lost_at = None
                        logger.info("[scan] face gone — resuming scan")
                        if self._on_face_lost:
                            threading.Thread(
                                target=self._on_face_lost, daemon=True
                            ).start()
                        self._scan_loop()
                else:
                    self._scan_loop()

            time.sleep(_POLL_INTERVAL)

    def _aim_at_face(self, u: int, v: int) -> None:
        """Use look_at_image() to point head at detected face pixel."""
        try:
            self._mini.look_at_image(u, v, duration=0.4, perform_movement=True)
        except Exception as e:
            logger.warning("[scan] look_at_image failed: %s", e)

    def _scan_loop(self) -> None:
        """One full sweep — checks for faces at each position."""
        from reachy_mini.utils import create_head_pose
        for yaw, pitch in _SCAN_POSES:
            if self._stop.is_set() or self._paused.is_set():
                return
            try:
                self._mini.goto_target(
                    head=create_head_pose(yaw=yaw, pitch=pitch, degrees=True),
                    duration=_SCAN_STEP_SEC,
                    method="ease_in_out",
                )
            except Exception as e:
                logger.warning("[scan] goto_target error: %s", e)
                return

            # Check for face throughout the move
            deadline = time.time() + _SCAN_STEP_SEC
            while time.time() < deadline:
                if self._stop.is_set() or self._paused.is_set():
                    return
                frame = self._mini.media.get_frame()
                if frame is not None:
                    u, v = self._tracker.detect(frame)
                    if u is not None:
                        logger.info("[scan] face found mid-sweep at (%d,%d)", u, v)
                        return   # outer loop handles it
                time.sleep(_POLL_INTERVAL)
