"""ADK tool: play a pre-recorded emotion on Reachy Mini."""

import time
import threading
import numpy as np

from reachy_mini.motion.recorded_move import RecordedMoves
import app.tools.present_slide as _ps
import app.robot.gestures as _ges

_RECORDED_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")

# Key emotions grouped for easy reference in the tool description
_EMOTION_GROUPS = {
    "happy": ["cheerful1", "enthusiastic1", "enthusiastic2", "laughing1", "laughing2", "proud1", "success1", "success2"],
    "curious": ["curious1", "inquiring1", "inquiring2", "inquiring3", "attentive1", "attentive2", "thoughtful1", "thoughtful2"],
    "sad": ["sad1", "sad2", "lonely1", "downcast1", "resigned1", "exhausted1", "tired1"],
    "surprised": ["surprised1", "surprised2", "amazed1", "oops1", "oops2"],
    "social": ["welcoming1", "welcoming2", "grateful1", "understanding1", "understanding2", "helpful1", "helpful2", "calming1"],
    "dance": ["dance1", "dance2", "dance3", "electric1"],
    "negative": ["irritated1", "irritated2", "displeased1", "displeased2", "frustrated1", "furious1", "rage1", "go_away1"],
}

_HZ = 50


def _play_emotion(emotion_name: str) -> None:
    idle = _ges._idle
    if idle:
        idle.pause()
    try:
        move = _RECORDED_MOVES.get(emotion_name)
        duration = move.duration
        t0 = time.monotonic()
        period = 1.0 / _HZ
        while True:
            t = time.monotonic() - t0
            if t >= duration:
                break
            head, antennas, _ = move.evaluate(t)
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


def play_emotion(emotion: str) -> str:
    """Play a pre-recorded emotion on Reachy Mini.

    Use this to express feelings naturally during conversation. Pick emotions that
    match the context — greet people with welcoming1, show curiosity with curious1,
    react to something funny with laughing1, etc.

    Key emotions available:
    - Happy/excited: cheerful1, enthusiastic1, laughing1, proud1, success1
    - Curious/thinking: curious1, inquiring1, thoughtful1, attentive1
    - Surprised: surprised1, amazed1, oops1
    - Social/warm: welcoming1, welcoming2, grateful1, calming1, helpful1
    - Sad/tired: sad1, lonely1, tired1, exhausted1
    - Dance-like: dance1, dance2, dance3, electric1
    - Negative: irritated1, displeased1, frustrated1, go_away1
    Full list: cheerful1, enthusiastic1, enthusiastic2, laughing1, laughing2, proud1,
    proud2, proud3, success1, success2, curious1, inquiring1, inquiring2, inquiring3,
    attentive1, attentive2, thoughtful1, thoughtful2, surprised1, surprised2, amazed1,
    oops1, oops2, welcoming1, welcoming2, grateful1, understanding1, understanding2,
    helpful1, helpful2, calming1, sad1, sad2, lonely1, downcast1, resigned1, tired1,
    exhausted1, irritated1, irritated2, displeased1, displeased2, frustrated1, furious1,
    rage1, go_away1, dance1, dance2, dance3, electric1, yes1, no1, shy1, sleep1,
    serenity1, relief1, relief2, confused1, uncomfortable1, anxious1, boredom1, boredom2

    Args:
        emotion: Name of the emotion to play (e.g. "cheerful1", "curious1").

    Returns:
        Confirmation or error message.
    """
    if not _ps._mini_ref:
        return "No robot connected."

    try:
        move = _RECORDED_MOVES.get(emotion)
    except Exception:
        available = _RECORDED_MOVES.list_moves()
        return f"Unknown emotion '{emotion}'. Available: {available}"

    threading.Thread(target=_play_emotion, args=(emotion,), daemon=True).start()
    return f"Playing emotion: {emotion}"
