"""Gesture sequences for slide transitions and emotional expression."""

import numpy as np
from reachy_mini.utils import create_head_pose

_NEUTRAL_HEAD = create_head_pose(pitch=0.0, roll=0.0, yaw=0.0)
_NEUTRAL_ANTENNAS = np.deg2rad([0.0, 0.0])


def slide_transition(mini) -> None:
    """Glance down briefly (checking notes) then look back up at audience."""
    mini.goto_target(
        head=create_head_pose(pitch=np.deg2rad(-20)),
        duration=0.5,
        method="minjerk",
    )
    mini.goto_target(
        head=_NEUTRAL_HEAD,
        duration=0.7,
        method="ease_in_out",
    )


def emotion_gesture(mini, emotion: str) -> None:
    """Brief expressive movement matching the slide's emotional tone."""
    if emotion == "excited":
        # Antennas up + slight head raise, then settle
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
        # Curious head tilt, then straighten
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
        # Slight downward look, antennas lower
        mini.goto_target(
            head=create_head_pose(pitch=np.deg2rad(-5)),
            antennas=np.deg2rad([-15, -15]),
            duration=0.6,
            method="minjerk",
        )
    else:  # neutral
        mini.goto_target(
            head=_NEUTRAL_HEAD,
            antennas=_NEUTRAL_ANTENNAS,
            duration=0.5,
        )
