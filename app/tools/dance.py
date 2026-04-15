"""ADK tool: play a dance move on Reachy Mini."""

import time
import threading
import numpy as np
from typing import Optional

from reachy_mini_dances_library.dance_move import DanceMove
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES

import app.tools.present_slide as _ps
import app.robot.gestures as _ges

_DANCE_NAMES = list(AVAILABLE_MOVES.keys())
_DANCE_DESCRIPTIONS = {
    name: AVAILABLE_MOVES[name][2].get("description", "") for name in _DANCE_NAMES
}

_HZ = 50  # control rate for dance playback


def _play_dance(move_name: str) -> None:
    """Run a dance move synchronously at _HZ, pausing idle behavior."""
    idle = _ges._idle
    if idle:
        idle.pause()
    try:
        dance = DanceMove(move_name)
        duration = dance.duration
        t0 = time.monotonic()
        period = 1.0 / _HZ
        while True:
            t = time.monotonic() - t0
            if t >= duration:
                break
            head, antennas, _ = dance.evaluate(t)
            if isinstance(antennas, tuple):
                antennas = np.array(antennas)
            try:
                _ps._mini_ref.set_target(head=head, antennas=antennas)
            except Exception:
                pass
            time.sleep(period)
    finally:
        if idle:
            idle.resume()


def dance(move: str = "random") -> str:
    """Play a dance move on Reachy Mini.

    Call this when someone asks Reachy to dance, move, or show off a specific move.
    Runs non-blocking in a background thread so conversation can continue.

    Available moves: simple_nod, head_tilt_roll, side_to_side_sway, dizzy_spin,
    stumble_and_recover, interwoven_spirals, sharp_side_tilt, side_peekaboo,
    yeah_nod, uh_huh_tilt, neck_recoil, chin_lead, groovy_sway_and_roll,
    chicken_peck, side_glance_flick, polyrhythm_combo, grid_snap,
    pendulum_swing, jackson_square

    Args:
        move: Name of the dance move, or "random" to pick one.

    Returns:
        Confirmation of which move is playing.
    """
    import random

    if not _ps._mini_ref:
        return "No robot connected."

    if move == "random" or move not in _DANCE_NAMES:
        move = random.choice(_DANCE_NAMES)

    threading.Thread(target=_play_dance, args=(move,), daemon=True).start()
    desc = _DANCE_DESCRIPTIONS.get(move, "")
    return f"Playing '{move}': {desc}"
