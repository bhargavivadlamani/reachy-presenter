"""Attention classifier — combines DOA, gaze, and VAD.

Output classes (matching sd_attention convention):
  SILENT      — no speech
  TO_HUMAN    — speech not directed at robot
  TO_COMPUTER — speech directed at robot

Decision logic:
  No VAD                                → SILENT
  VAD + frontal face + DOA toward robot → TO_COMPUTER  (strong signal)
  VAD + frontal face + DOA away         → TO_HUMAN     (facing robot but sound from side)
  VAD + no frontal + DOA toward robot   → TO_COMPUTER  (sound toward robot, face not visible)
  VAD + no frontal + DOA away           → TO_HUMAN
  VAD + no face detected                → TO_COMPUTER  (assume direct address if no crowd)

Smoothing: majority vote over last N frames prevents rapid flickering.
"""

import logging
import threading
import time
from enum import IntEnum
from typing import Callable, Optional

from app.robot.doa import DOAEstimator, VAD_RMS_MIN
from app.robot.gaze import estimate_gaze, GazeResult

logger = logging.getLogger(__name__)

DOA_FRONTAL_RANGE_DEG = 50.0   # ±50° counts as "sound toward robot"
SMOOTHING_FRAMES      = 6      # frames for majority vote
FRAME_INTERVAL        = 0.1    # seconds between frames (~10 fps)


class AttentionClass(IntEnum):
    SILENT      = 0
    TO_HUMAN    = 1
    TO_COMPUTER = 2


class AttentionClassifier:
    """Runs continuously in background thread, fires callback on class change."""

    def __init__(
        self,
        mini,
        on_change: Optional[Callable[[AttentionClass], None]] = None,
    ) -> None:
        self._mini = mini
        self._on_change = on_change
        self._doa = DOAEstimator()
        self._current = AttentionClass.SILENT
        self._history: list[AttentionClass] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._doa.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("[attention] classifier started")

    def stop(self) -> None:
        self._stop.set()
        self._doa.stop()

    @property
    def current(self) -> AttentionClass:
        return self._current

    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._mini.media.get_frame()
                if frame is None:
                    time.sleep(FRAME_INTERVAL)
                    continue

                gaze: GazeResult = estimate_gaze(frame)
                doa_angle: float = self._doa.angle
                vad_active: bool = self._doa.last_rms > VAD_RMS_MIN

                cls = self._classify(vad_active, gaze, doa_angle)
                self._smooth_and_emit(cls)

            except Exception as e:
                logger.debug("[attention] loop error: %s", e)

            time.sleep(FRAME_INTERVAL)

    def _classify(
        self,
        vad_active: bool,
        gaze: GazeResult,
        doa_angle: float,
    ) -> AttentionClass:
        if not vad_active:
            return AttentionClass.SILENT

        doa_toward = abs(doa_angle) < DOA_FRONTAL_RANGE_DEG

        if gaze.is_frontal and doa_toward:
            return AttentionClass.TO_COMPUTER
        if gaze.is_frontal and not doa_toward:
            return AttentionClass.TO_HUMAN
        if doa_toward:
            return AttentionClass.TO_COMPUTER
        return AttentionClass.TO_HUMAN

    def _smooth_and_emit(self, cls: AttentionClass) -> None:
        self._history.append(cls)
        if len(self._history) > SMOOTHING_FRAMES:
            self._history.pop(0)

        smoothed = max(set(self._history), key=self._history.count)
        if smoothed != self._current:
            self._current = smoothed
            logger.info(
                "[attention] %s (doa=%.1f° vad=%s)",
                smoothed.name,
                self._doa.angle,
                self._doa.last_rms > VAD_RMS_MIN,
            )
            if self._on_change:
                self._on_change(smoothed)
