"""Lightweight idle behavior: breathing head bob + antenna sway while Reachy is listening."""

import threading
import time

import numpy as np
from numpy.typing import NDArray
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import linear_pose_interpolation

_NEUTRAL_HEAD = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
_NEUTRAL_ANTENNAS = np.array([-0.1745, 0.1745])  # ~10° resting offset


def _breathing_evaluate(t: float, start_pose: NDArray, start_antennas: NDArray):
    """Return (head_pose, antennas) for time t into the breathing cycle."""
    interp_duration = 1.0  # blend to neutral over first second

    if t < interp_duration:
        alpha = t / interp_duration
        head = linear_pose_interpolation(start_pose, _NEUTRAL_HEAD, alpha)
        antennas = (1 - alpha) * start_antennas + alpha * _NEUTRAL_ANTENNAS
    else:
        bt = t - interp_duration
        z = 0.005 * np.sin(2 * np.pi * 0.1 * bt)          # gentle 6 breaths/min
        head = create_head_pose(x=0, y=0, z=z, roll=0, pitch=0, yaw=0, degrees=True, mm=False)
        sway = np.deg2rad(15) * np.sin(2 * np.pi * 0.5 * bt)  # antenna sway
        antennas = np.array([sway, -sway])

    return head, antennas.astype(np.float64)


class IdleBehavior:
    """Background thread that animates breathing + antenna sway during idle.

    Usage:
        idle = IdleBehavior(mini)
        idle.start()
        # ... before a gesture ...
        idle.pause()
        mini.goto_target(...)
        idle.resume()
        # ... on shutdown ...
        idle.stop()
    """

    _HZ = 20  # control rate — low enough to not stress the Pi

    def __init__(self, mini: ReachyMini) -> None:
        self.mini = mini
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # set → paused
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="idle-behavior")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def pause(self) -> None:
        """Freeze idle animation (call before goto_target gestures)."""
        self._pause_event.set()
        time.sleep(1.0 / self._HZ + 0.01)  # wait for current tick to finish

    def resume(self) -> None:
        """Resume idle animation after a gesture."""
        self._pause_event.clear()

    def _run(self) -> None:
        period = 1.0 / self._HZ
        start_pose = _NEUTRAL_HEAD
        start_antennas = _NEUTRAL_ANTENNAS.copy()
        t0 = time.monotonic()

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                # Reset clock so animation restarts smoothly after resume
                t0 = time.monotonic()
                start_pose = _NEUTRAL_HEAD
                start_antennas = _NEUTRAL_ANTENNAS.copy()
                time.sleep(period)
                continue

            t = time.monotonic() - t0
            head, antennas = _breathing_evaluate(t, start_pose, start_antennas)
            try:
                self.mini.set_target(head=head, antennas=antennas)
            except Exception:
                pass  # robot not ready yet, skip tick
            time.sleep(period)
