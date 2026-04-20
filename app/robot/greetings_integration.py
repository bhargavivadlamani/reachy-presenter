"""Greetings integration: face tracking + hand gesture reactions.

Runs a background thread that:
1. Continuously reads camera frames
2. Detects faces → feeds pose to IdleBehavior so Reachy looks at you
3. Detects hand gestures → plays matching emotions (wave, thumbs up/down, point)

The idle behavior thread is the sole caller of set_target, so there are no
conflicts: this module just updates idle's face_pose target.

Gesture reactions pause/resume idle internally (same as play_emotion does).
The integration skips all processing when the agent is busy (idle is paused).
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_GESTURE_COOLDOWN = 4.0  # seconds between gesture reactions

_GESTURE_TO_EMOTION = {
    "Open_Palm": "welcoming1",   # wave hello
    "Thumb_Up": "proud1",        # thumbs up
    "Thumb_Down": "sad2",        # thumbs down
    "Pointing_Up": "attentive2", # pointing
}


class GreetingsIntegration:
    """Background face tracking and hand gesture recognition.

    Face tracking is handed off to IdleBehavior via update_face_pose() so
    only one thread ever calls set_target on the robot.

    Gesture reactions use the existing play_emotion infrastructure (which
    already handles idle pause/resume correctly).
    """

    def __init__(self, mini, idle) -> None:
        self._mini = mini
        self._idle = idle
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_gesture_time: float = 0.0

        # Set up in _init_trackers (called inside the background thread)
        self._camera = None
        self._head_tracker = None
        self._palm_tracker = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="greetings-integration"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_trackers(self) -> bool:
        """Initialize camera + face/gesture trackers. Returns False on failure."""
        try:
            from reachy_mini_greetings.camera_manager import CameraManager
            from reachy_mini_greetings.head_tracker import HeadTracker
            from reachy_mini_greetings.palm_tracker import PalmTracker

            self._camera = CameraManager(self._mini)

            # HeadTracker needs camera_manager for focal lengths.
            # We call get_head_position() directly, so movement_manager=None is safe.
            self._head_tracker = HeadTracker(
                camera_manager=self._camera,
                movement_manager=None,  # not using track_head()
                yolo_model=False,
            )

            # PalmTracker: movement_manager only used in react_to_hand_gesture(),
            # which we don't call — so None is fine here too.
            self._palm_tracker = PalmTracker(movement_manager=None)

            logger.info("GreetingsIntegration: trackers ready.")
            return True

        except Exception as e:
            logger.warning(f"GreetingsIntegration unavailable: {e}")
            return False

    def _pose_in_bounds(self, pose, pitch_threshold=25, yaw_threshold=45) -> bool:
        """True if the head pose is within a comfortable tracking range."""
        try:
            from scipy.spatial.transform import Rotation as R
            euler = R.from_matrix(pose[0:3, 0:3]).as_euler("xyz", degrees=True)
            return abs(euler[1]) <= pitch_threshold and abs(euler[2]) <= yaw_threshold
        except Exception:
            return False

    def _react_to_gesture(self, gesture: str) -> None:
        """Play an emotion matching the detected hand gesture (runs in its own thread)."""
        from app.tools.play_emotion import _play_emotion
        emotion = _GESTURE_TO_EMOTION.get(gesture, "curious1")
        logger.info(f"Gesture '{gesture}' → emotion '{emotion}'")
        _play_emotion(emotion)  # pause/resume idle internally

    def _run(self) -> None:
        if not self._init_trackers():
            return

        logger.info("GreetingsIntegration: running.")

        while not self._stop_event.is_set():
            # Skip processing if camera has no frame yet
            if not self._camera.frame:
                time.sleep(0.05)
                continue

            frame = self._camera.frame[0]

            # Only act when agent is not speaking / doing a gesture
            agent_busy = self._idle._pause_event.is_set()

            if not agent_busy:
                # --- Face tracking ---
                try:
                    eye_center, _ = self._head_tracker.get_head_position(frame)
                    if eye_center is not None:
                        pose = self._head_tracker.pose_from_head_uv(
                            eye_center[0], eye_center[1], self._mini
                        )
                        if self._pose_in_bounds(pose):
                            self._idle.update_face_pose(pose)
                except Exception as e:
                    logger.debug(f"Face tracking error: {e}")

                # --- Hand gesture detection (with cooldown) ---
                if time.time() - self._last_gesture_time > _GESTURE_COOLDOWN:
                    try:
                        gesture = self._palm_tracker.gesture_detected(frame)
                        if gesture:
                            self._last_gesture_time = time.time()
                            threading.Thread(
                                target=self._react_to_gesture,
                                args=(gesture,),
                                daemon=True,
                                name=f"gesture-{gesture}",
                            ).start()
                    except Exception as e:
                        logger.debug(f"Gesture detection error: {e}")

            time.sleep(0.05)  # ~20 fps — matches idle behavior control rate
