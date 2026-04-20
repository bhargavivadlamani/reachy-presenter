"""Expressive gesture sequences for the audience robot."""

import numpy as np
from reachy_mini.utils import create_head_pose

_NEUTRAL_HEAD = create_head_pose(pitch=0.0, roll=0.0, yaw=0.0)
_NEUTRAL_ANTENNAS = np.deg2rad([0.0, 0.0])


def emotion_gesture(mini, emotion: str) -> None:
    """Brief expressive movement matching the given emotion."""
    if emotion == "excited":
        mini.goto_target(
            head=create_head_pose(pitch=np.deg2rad(8)),
            antennas=np.deg2rad([40, 40]),
            duration=0.5,
            method="cartoon",
        )
        mini.goto_target(
            head=_NEUTRAL_HEAD,
            antennas=_NEUTRAL_ANTENNAS,
            duration=0.4,
        )
    elif emotion == "questioning":
        mini.goto_target(
            head=create_head_pose(roll=np.deg2rad(12)),
            duration=0.6,
            method="ease_in_out",
        )
        mini.goto_target(
            head=_NEUTRAL_HEAD,
            duration=0.5,
            method="ease_in_out",
        )
    elif emotion == "serious":
        mini.goto_target(
            head=create_head_pose(pitch=np.deg2rad(-5)),
            antennas=np.deg2rad([-15, -15]),
            duration=0.6,
            method="minjerk",
        )
    else:
        mini.goto_target(
            head=_NEUTRAL_HEAD,
            antennas=_NEUTRAL_ANTENNAS,
            duration=0.5,
        )
