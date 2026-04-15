"""ADK tool: move Reachy Mini's head in a direction."""

import numpy as np
from reachy_mini.utils import create_head_pose
import app.tools.present_slide as _ps
import app.robot.gestures as _ges

_DIRECTIONS = {
    "left":  create_head_pose(yaw=np.deg2rad(40)),
    "right": create_head_pose(yaw=np.deg2rad(-40)),
    "up":    create_head_pose(pitch=np.deg2rad(-30)),
    "down":  create_head_pose(pitch=np.deg2rad(30)),
    "front": create_head_pose(0, 0, 0, 0, 0, 0, degrees=True),
}


def move_head(direction: str) -> str:
    """Move Reachy Mini's head to look in a direction.

    Use this to make eye contact, look at something, nod toward someone,
    or react physically to what's being said. Always return to "front" after
    looking away.

    Args:
        direction: One of "left", "right", "up", "down", "front".

    Returns:
        Confirmation message.
    """
    if not _ps._mini_ref:
        return "No robot connected."

    direction = direction.lower().strip()
    target = _DIRECTIONS.get(direction)
    if target is None:
        return f"Unknown direction '{direction}'. Use: left, right, up, down, front."

    if _ges._idle:
        _ges._idle.pause()
    try:
        _ps._mini_ref.goto_target(head=target, duration=0.5, method="ease_in_out")
    finally:
        if _ges._idle:
            _ges._idle.resume()

    return f"Looking {direction}."
