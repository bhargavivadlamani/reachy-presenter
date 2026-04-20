"""AttentiveListener — natural gaze + antenna behavior during conversation.

Pauses ScanBehavior and takes over head/antenna control whenever speech is
detected. Behaviour:

  Robot 2 speaking  → look toward Robot 1's configured position (ROBOT1_YAW_DEG)
  Someone else speaking → look toward sound source via DOA angle
  Antennas           → slow oscillation while speech is detected (engaged look)
  Silence (>1s)     → return antennas to neutral, resume ScanBehavior

Set ROBOT1_YAW_DEG in robot 2's .env to match your physical booth layout.
Negative = Robot 1 is to the left of Robot 2 when facing forward.
"""

import logging
import math
import os
import threading
import time
from typing import Optional

import numpy as np
from reachy_mini.utils import create_head_pose

logger = logging.getLogger(__name__)

_SPEECH_THRESHOLD   = 0.012  # RMS above this = someone speaking
_SILENCE_HOLD       = 1.2    # seconds of quiet before handing back to scanner
_HEAD_DURATION      = 0.6    # head movement duration (s) — slower = smoother
_HEAD_DURATION_IDLE = 1.0    # duration for look-at-Robot1 during idle (very smooth)
_POLL               = 0.3    # loop interval — longer than duration to avoid command flood
_ANT_PERIOD         = 1.8    # antenna oscillation period (s)
_ANT_AMP_DEG        = 10     # antenna oscillation amplitude (degrees)
_ROBOT1_LOOK_INTERVAL = 3.0  # seconds between "look at Robot 1" nudges during idle

_ROBOT1_YAW_DEG   = float(os.environ.get("ROBOT1_YAW_DEG", "-35"))
_ROBOT1_PITCH_DEG = -5.0    # slight downward look toward Robot 1


class AttentiveListener:
    """Controls Robot 2's gaze and antennas based on who is speaking.

    Wire-up:
        listener = AttentiveListener(mini, doa, scanner)
        listener.start()
        # before/after Robot 2 speaks:
        listener.set_speaking(True)
        agent._speak(text)
        listener.set_speaking(False)
    """

    def __init__(self, mini, doa, scanner) -> None:
        self._mini = mini
        self._doa = doa
        self._scanner = scanner
        self._stop = threading.Event()
        self._speaking = False          # True while Robot 2's TTS is playing
        self._visitor_present = False   # True while a visitor is at the booth
        self._scanner_paused = False    # shared flag so _run tracks external pauses
        self._t = 0.0                   # phase counter for antenna oscillation
        self._last_robot1_look = 0.0    # timestamp of last "look at Robot 1" nudge

    def set_speaking(self, is_speaking: bool) -> None:
        self._speaking = is_speaking

    def set_visitor_present(self, present: bool) -> None:
        """Call when a visitor arrives/leaves. Pauses scanner and smoothly faces Robot 1."""
        self._visitor_present = present
        if present and not self._scanner_paused:
            self._scanner.pause()
            self._scanner_paused = True
            self._look_at_yaw(_ROBOT1_YAW_DEG, _ROBOT1_PITCH_DEG, duration=1.2)
            self._last_robot1_look = time.time()

    def start(self) -> None:
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        logger.info("[attentive] started — robot1_yaw=%.0f°", _ROBOT1_YAW_DEG)

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------

    def _look_at_yaw(self, yaw_deg: float, pitch_deg: float = -8.0,
                     duration: float = _HEAD_DURATION) -> None:
        try:
            self._mini.goto_target(
                head=create_head_pose(yaw=yaw_deg, pitch=pitch_deg, degrees=True),
                duration=duration,
                method="ease_in_out",
            )
        except Exception as e:
            logger.debug("[attentive] head move error: %s", e)

    def _set_antennas(self, left_deg: float, right_deg: float) -> None:
        try:
            self._mini.goto_target(
                antennas=np.deg2rad([left_deg, right_deg]),
                duration=_POLL,
            )
        except Exception as e:
            logger.debug("[attentive] antenna error: %s", e)

    def _run(self) -> None:
        silent_since: Optional[float] = None

        while not self._stop.is_set():
            rms = self._doa.last_rms
            active = self._speaking or rms >= _SPEECH_THRESHOLD

            if active:
                silent_since = None

                if not self._scanner_paused:
                    self._scanner.pause()
                    self._scanner_paused = True

                if self._speaking:
                    # Robot 2 is talking — face toward Robot 1
                    self._look_at_yaw(_ROBOT1_YAW_DEG, _ROBOT1_PITCH_DEG)
                else:
                    # Someone else is talking — look toward them via DOA
                    yaw = float(np.clip(self._doa.angle, -55, 55))
                    self._look_at_yaw(yaw)

                # Antenna oscillation — natural engaged/listening motion
                self._t += _POLL
                amp = _ANT_AMP_DEG * math.sin(2 * math.pi * self._t / _ANT_PERIOD)
                self._set_antennas(amp, -amp)

            else:
                # Silence — hold on Robot 1 if visitor present, otherwise resume scanner
                if self._scanner_paused:
                    if silent_since is None:
                        silent_since = time.time()
                    elif time.time() - silent_since >= _SILENCE_HOLD:
                        self._set_antennas(0.0, 0.0)
                        silent_since = None
                        if self._visitor_present:
                            # Nudge toward Robot 1 — but only every few seconds
                            # so we don't flood goto_target commands
                            now = time.time()
                            if now - self._last_robot1_look >= _ROBOT1_LOOK_INTERVAL:
                                self._look_at_yaw(_ROBOT1_YAW_DEG, _ROBOT1_PITCH_DEG,
                                                  duration=_HEAD_DURATION_IDLE)
                                self._last_robot1_look = now
                        else:
                            self._scanner.resume()
                            self._scanner_paused = False

            time.sleep(_POLL)
