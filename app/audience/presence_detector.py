"""Presence detector — robot scans around until it sees a person at the booth.

Uses ScanBehavior to physically move the head and camera. When a face is
detected and stays visible for DWELL_SECONDS, fires on_arrived(). When all
faces disappear for GONE_SECONDS, fires on_left() and resumes scanning.
"""

import logging
import os
import threading
import time
from typing import Callable, Optional

from app.robot.scan_behavior import ScanBehavior

logger = logging.getLogger(__name__)

DWELL_SECONDS = float(os.environ.get("PRESENCE_DWELL_SECONDS", "3.0"))
GONE_SECONDS  = float(os.environ.get("PRESENCE_GONE_SECONDS", "20.0"))


class PresenceDetector:
    """Wraps ScanBehavior — fires callbacks when people arrive/leave."""

    def __init__(
        self,
        mini,
        on_arrived: Optional[Callable[[], None]] = None,
        on_left:    Optional[Callable[[], None]] = None,
    ) -> None:
        self._mini = mini
        self._on_arrived = on_arrived
        self._on_left = on_left
        self.person_present = False
        self._face_since: Optional[float] = None
        self._dwell_timer: Optional[threading.Timer] = None

        self._scanner = ScanBehavior(
            mini=mini,
            on_face_found=self._handle_face_found,
            on_face_lost=self._handle_face_lost,
        )

    @property
    def scanner(self) -> ScanBehavior:
        return self._scanner

    def start(self) -> None:
        self._scanner.start()
        logger.info("[presence] detector started (dwell=%.1fs gone=%.1fs)",
                    DWELL_SECONDS, GONE_SECONDS)

    def stop(self) -> None:
        self._scanner.stop()
        if self._dwell_timer:
            self._dwell_timer.cancel()

    def pause_scan(self) -> None:
        self._scanner.pause()

    def resume_scan(self) -> None:
        self._scanner.resume()

    # ------------------------------------------------------------------

    def _handle_face_found(self) -> None:
        """Called by ScanBehavior when a face is detected."""
        if self.person_present:
            return
        if self._face_since is None:
            self._face_since = time.time()
            logger.debug("[presence] face seen — starting dwell timer")
            # Fire arrived after dwell period
            self._dwell_timer = threading.Timer(DWELL_SECONDS, self._confirm_arrived)
            self._dwell_timer.start()

    def _confirm_arrived(self) -> None:
        """Called after face has been visible for DWELL_SECONDS."""
        if not self._scanner.face_visible:
            self._face_since = None
            return
        self.person_present = True
        logger.info("[presence] person ARRIVED")
        if self._on_arrived:
            threading.Thread(target=self._on_arrived, daemon=True).start()

    def _handle_face_lost(self) -> None:
        """Called by ScanBehavior when face disappears for GONE_SECONDS."""
        self._face_since = None
        if self._dwell_timer:
            self._dwell_timer.cancel()
            self._dwell_timer = None
        if self.person_present:
            self.person_present = False
            logger.info("[presence] person LEFT — resuming scan")
            if self._on_left:
                threading.Thread(target=self._on_left, daemon=True).start()
