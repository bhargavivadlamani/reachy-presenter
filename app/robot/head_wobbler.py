"""HeadWobbler — moves Reachy's head in sync with audio rhythm while speaking.

Adapted from reachy_mini_conversation_app.audio.head_wobbler.
Instead of MovementManager secondary offsets, applies movements directly
via mini.goto_target() — pauses IdleBehavior for the duration.
"""

import base64
import logging
import queue
import threading
import time
from typing import Optional

import numpy as np
from reachy_mini.utils import create_head_pose

from app.robot.speech_tapper import HOP_MS, SwayRollRT

SAMPLE_RATE = 24000
MOVEMENT_LATENCY_S = 0.2

logger = logging.getLogger(__name__)


class HeadWobbler:
    """Feeds Gemini audio output into SwayRollRT and moves the head with it."""

    def __init__(self, mini) -> None:
        self._mini = mini
        self.audio_queue: "queue.Queue" = queue.Queue()
        self.sway = SwayRollRT()
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._sway_lock = threading.Lock()
        self._generation = 0
        self._base_ts: Optional[float] = None
        self._hops_done: int = 0
        self._thread: Optional[threading.Thread] = None

    def feed(self, delta_b64: str) -> None:
        """Push a base64-encoded int16 PCM chunk into the queue."""
        buf = np.frombuffer(base64.b64decode(delta_b64), dtype=np.int16).reshape(1, -1)
        with self._state_lock:
            gen = self._generation
        self.audio_queue.put((gen, SAMPLE_RATE, buf))

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def reset(self) -> None:
        """Call when a Gemini turn ends to drain stale audio."""
        with self._state_lock:
            self._generation += 1
            self._base_ts = None
            self._hops_done = 0
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        with self._sway_lock:
            self.sway.reset()

    # ------------------------------------------------------------------

    def _apply(self, pitch_rad: float, yaw_rad: float, roll_rad: float) -> None:
        try:
            self._mini.goto_target(
                head=create_head_pose(pitch=pitch_rad, yaw=yaw_rad, roll=roll_rad),
                duration=HOP_MS / 1000.0,
                method="linear",
            )
        except Exception:
            pass

    def _loop(self) -> None:
        hop_dt = HOP_MS / 1000.0

        while not self._stop_event.is_set():
            try:
                chunk_gen, sr, chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                time.sleep(MOVEMENT_LATENCY_S)
                continue

            try:
                with self._state_lock:
                    cur_gen = self._generation
                if chunk_gen != cur_gen:
                    continue

                if self._base_ts is None:
                    with self._state_lock:
                        if self._base_ts is None:
                            self._base_ts = time.monotonic()

                pcm = np.asarray(chunk).squeeze(0)
                with self._sway_lock:
                    results = self.sway.feed(pcm, sr)

                i = 0
                while i < len(results):
                    with self._state_lock:
                        if self._generation != chunk_gen:
                            break
                        base_ts = self._base_ts
                        hops_done = self._hops_done

                    if base_ts is None:
                        base_ts = time.monotonic()
                        with self._state_lock:
                            if self._base_ts is None:
                                self._base_ts = base_ts

                    target = base_ts + MOVEMENT_LATENCY_S + hops_done * hop_dt
                    now = time.monotonic()

                    if now - target >= hop_dt:
                        lag = int((now - target) / hop_dt)
                        drop = min(lag, len(results) - i - 1)
                        if drop > 0:
                            with self._state_lock:
                                self._hops_done += drop
                            i += drop
                            continue

                    if target > now:
                        time.sleep(target - now)
                        with self._state_lock:
                            if self._generation != chunk_gen:
                                break

                    r = results[i]
                    self._apply(r["pitch_rad"], r["yaw_rad"], r["roll_rad"])

                    with self._state_lock:
                        self._hops_done += 1
                    i += 1
            finally:
                self.audio_queue.task_done()
