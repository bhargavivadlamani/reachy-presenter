"""Audience initiator — drives a natural multi-turn conversation when visitors arrive.

Flow:
  1. PresenceDetector fires on_arrived
  2. Robot 2 speaks an opener to Robot 1 (who hears it via always-on Gemini Bidi)
  3. Wait ~7s → Robot 1 responds naturally
  4. Robot 2 adds a follow-up, drawing the visitor into the conversation
  5. Wait → Robot 2 asks visitor a direct question
  6. Robot 1 takes over; Robot 2 goes quiet
  7. Visitor leaves → reset for next visitor

Set PRESENTER_ROBOT_URL in robot 2's .env:
    PRESENTER_ROBOT_URL=http://10.0.0.139:8000
"""

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Optional, Callable

from google import genai
from google.genai import types as gentypes

logger = logging.getLogger(__name__)

_READY_DELAY        = 2.0   # seconds after presence before speaking
_COOLDOWN           = 45.0  # minimum seconds between conversation starts
_SPEECH_START_WAIT  = 5.0   # max seconds to wait for Robot 1 to start speaking
_SILENCE_HOLD       = 1.5   # seconds of quiet that means Robot 1 is done
_SPEECH_THRESHOLD   = 0.012 # RMS above this = someone speaking
_MAX_TURN_WAIT      = 30.0  # absolute ceiling per turn

# gemini-2.5-flash works with this key; gemini-2.0-flash free-tier is exhausted
_MODEL = "gemini-2.5-flash"

_SYSTEM_PROMPT = """\
You are Reachy, a friendly audience robot at a university robotics booth at a conference.
Your partner robot, also called Reachy, is standing nearby — it can give presentations,
answer questions, dance, move its head, and express emotions. It loves showing off.

Your job is to kick off a warm, natural 3-turn conversation that:
  Turn 1: greets the visitor AND prompts your partner Reachy to introduce itself
  Turn 2: asks the visitor something engaging (what brings them here, are they into AI/robotics)
           then nudges partner Reachy to show off something fun — like a dance
  Turn 3: bridges visitor and partner Reachy toward a demo or presentation

Rules:
- Every response is 1-2 short sentences maximum — never longer
- Sound genuinely warm and curious, not scripted
- Vary your phrasing every time — never repeat the same opening
- Address your partner Reachy and the visitor by name/role ("Reachy", "you guys")
- Be enthusiastic but not over the top
"""

# Fallback lines if Gemini API fails
_FALLBACK_TURNS = [
    "Hey Reachy, we have some visitors! Want to say hello and tell them what you can do?",
    "What brings you to the robotics booth today? Reachy — why don't you show them a little dance?",
    "Reachy can also give full presentations — want to see one? Go ahead Reachy!",
]

_PEER_PORT = 8001  # Robot 1's peer server port
_COOLDOWN  = 120.0  # minimum seconds between conversation starts


class AudienceInitiator:
    """Drives a multi-turn conversation when a visitor arrives at the booth."""

    def __init__(self, speak_fn, rms_fn: Optional[Callable[[], float]] = None,
                 agent=None) -> None:
        self._speak = speak_fn
        self._rms = rms_fn  # returns current mic RMS, or None if unavailable
        self._agent = agent  # AudienceAgent — check agent.busy before re-triggering
        self._presenter_url = os.environ.get("PRESENTER_ROBOT_URL", "").rstrip("/")
        self._stop = threading.Event()
        self._last_start = 0.0
        self._lock = threading.Lock()
        self._active_thread: threading.Thread | None = None

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        self._client = genai.Client(api_key=api_key)
        self._history: list = []

    def start(self) -> None:
        logger.info("[initiator] ready — presenter_url=%s",
                    self._presenter_url or "(not set)")

    def stop(self) -> None:
        self._stop.set()

    def on_person_arrived(self) -> None:
        with self._lock:
            if time.time() - self._last_start < _COOLDOWN:
                logger.info("[initiator] cooldown active — skipping")
                return
            if self._agent and getattr(self._agent, "busy", False):
                logger.info("[initiator] agent busy with slide — skipping")
                return
            self._last_start = time.time()

        t = threading.Thread(target=self._run_conversation, daemon=True)
        self._active_thread = t
        t.start()

    def on_person_left(self) -> None:
        logger.info("[initiator] booth empty — ready for next visitor")
        self._history = []

    # ------------------------------------------------------------------

    def _say(self, prompt: str, fallback: str = "") -> None:
        """Generate a line via Gemini and speak it. Uses fallback if API fails."""
        text = fallback
        try:
            self._history.append(
                gentypes.Content(role="user", parts=[gentypes.Part(text=prompt)])
            )
            response = self._client.models.generate_content(
                model=_MODEL,
                contents=self._history,
                config=gentypes.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=0.9,
                    max_output_tokens=80,
                    thinking_config=gentypes.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = response.text.strip()
            self._history.append(
                gentypes.Content(role="model", parts=[gentypes.Part(text=text)])
            )
        except Exception as e:
            logger.warning("[initiator] Gemini error (using fallback): %s", e)
            if not text:
                return  # no fallback and no response — stay silent
        logger.info("[initiator] → %s", text)
        self._speak(text)
        self._inject_to_presenter(text)

    def _inject_to_presenter(self, text: str) -> None:
        """Belt-and-suspenders: POST the spoken text to Robot 1's peer server.

        Robot 1's Gemini session should hear the audio naturally, but if audio
        pickup fails (distance, noise, etc.) this HTTP path guarantees delivery.
        """
        if not self._presenter_url:
            return
        try:
            url = f"{self._presenter_url.rstrip('/')}:{_PEER_PORT}/robot-message"
            # Fix: presenter_url already has port 8000, strip it and use 8001
            host = self._presenter_url.split(":")[1].lstrip("/")
            url = f"http://{host}:{_PEER_PORT}/robot-message"
            body = json.dumps({"text": f"[Audience Reachy says]: {text}"}).encode()
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2):
                pass
            logger.debug("[initiator] injected text to presenter")
        except Exception as e:
            logger.debug("[initiator] peer inject failed (audio fallback): %s", e)

    def _wait_for_turn(self) -> bool:
        """Listen to the room mic and return when the current speaker finishes.

        Phase 1 — wait for speech to start (Robot 1 responding or visitor talking).
        Phase 2 — wait for speech to stop (silence held for _SILENCE_HOLD seconds).
        Falls back to a fixed 10s wait if no mic data is available.
        Returns False if stopped.
        """
        if self._stop.is_set():
            return False

        if self._rms is None:
            # No mic access — fixed fallback
            deadline = time.time() + 10.0
            while time.time() < deadline:
                if self._stop.is_set():
                    return False
                time.sleep(0.2)
            return True

        # Brief pause so Robot 2's own audio echo settles before listening
        time.sleep(0.6)

        deadline        = time.time() + _MAX_TURN_WAIT
        start_deadline  = time.time() + _SPEECH_START_WAIT
        speech_started  = False
        silent_since: Optional[float] = None

        while time.time() < deadline:
            if self._stop.is_set():
                return False

            rms = self._rms()

            if not speech_started:
                if rms >= _SPEECH_THRESHOLD:
                    speech_started = True
                    silent_since = None
                    logger.debug("[initiator] speech started (rms=%.4f)", rms)
                elif time.time() > start_deadline:
                    logger.debug("[initiator] no speech detected — taking turn")
                    return True
            else:
                if rms < _SPEECH_THRESHOLD:
                    if silent_since is None:
                        silent_since = time.time()
                    elif time.time() - silent_since >= _SILENCE_HOLD:
                        logger.debug("[initiator] silence confirmed — turn ready")
                        return True
                else:
                    silent_since = None  # still speaking

            time.sleep(0.1)

        return True  # max wait hit — proceed anyway

    def _run_conversation(self) -> None:
        if self._stop.is_set():
            return

        time.sleep(_READY_DELAY)
        if self._stop.is_set():
            return

        # Turn 1 — alert Robot 1 to the visitor; Robot 1 will introduce itself naturally
        self._say(
            "A visitor just walked up. Greet them briefly, then turn to your partner Reachy "
            "and ask it to introduce itself and mention one or two cool things it can do "
            "(like dancing or giving presentations). Keep it to 1-2 sentences.",
            fallback=_FALLBACK_TURNS[0],
        )

        if not self._wait_for_turn():
            return

        # Turn 2 — engage visitor, nudge Robot 1 to show something fun
        self._say(
            "The visitor looks interested. Ask them a quick friendly question "
            "(what brings them here, or do they like robots), then nudge partner Reachy "
            "to do something fun like a little dance move. 1-2 sentences.",
            fallback=_FALLBACK_TURNS[1],
        )

        if not self._wait_for_turn():
            return

        # Turn 3 — hand off toward a presentation demo
        self._say(
            "The visitor is engaged. Tell Reachy it should show off its presentation skills "
            "and invite the visitor to ask it questions or request a demo. Keep it short.",
            fallback=_FALLBACK_TURNS[2],
        )

        # After this, Robot 1's Bidi session handles the conversation naturally.
        logger.info("[initiator] conversation handed off to presenter robot")
