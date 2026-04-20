"""Audience agent — second Reachy Mini that reacts and asks questions.

Architecture:
- Physical reactions via direct gestures (emotion_gesture from app.robot.gestures)
- Question generation via Gemini Flash text (lightweight, not Bidi)
- Speech output via Gemini TTS → push_audio_sample on audience robot
- Timing: wait is estimated from script word count so audience speaks AFTER presenter

Flow per slide:
  1. Presenter calls present_slide() → notifier POSTs {slide_number, script, word_count}
  2. Audience does an excited reaction immediately (presenter is starting)
  3. Audience waits estimated speaking duration + buffer
  4. Audience generates a question and speaks it
  5. Presenter's always-on mic hears the question and Gemini responds
"""

import logging
import threading
import time
import os
import numpy as np
from typing import Optional, Callable

from google import genai
from google.genai import types
from scipy.signal import resample as scipy_resample
from reachy_mini import ReachyMini

from app.audio_helpers import GEM_SR, decode_pcm

logger = logging.getLogger(__name__)

_WORDS_PER_SECOND = 2.8   # average speaking rate — tune if presenter is faster/slower
_MIN_WAIT = 4.0            # minimum seconds to wait even for very short scripts
_POST_SPEECH_BUFFER = 1.5  # extra seconds after estimated duration before audience speaks

_AUDIENCE_SYSTEM_PROMPT = """You are a curious, engaged audience member watching a robot give a presentation.
You ask short, natural questions (1-2 sentences max) about what was just presented.
Be genuinely curious — ask about specifics, implications, or things you didn't fully understand.
Never repeat the same question. Never ask generic questions like "can you tell me more?".
Respond with ONLY the question text, nothing else."""


class AudienceAgent:
    """Controls the audience Reachy Mini — reactions + voice questions.

    Usage:
        audience = AudienceAgent(audience_mini)
        audience.start()
        # When a slide event arrives:
        audience.on_slide_presented(slide_number, script_text)
        # On shutdown:
        audience.stop()
    """

    def __init__(
        self,
        audience_mini: ReachyMini,
        presenter_mini: ReachyMini = None,   # kept for API compat, unused
        mute_presenter_mic: Optional[Callable] = None,
        unmute_presenter_mic: Optional[Callable] = None,
        on_speaking_changed: Optional[Callable[[bool], None]] = None,
    ) -> None:
        self.audience_mini = audience_mini
        self._mute_presenter = mute_presenter_mic or (lambda: None)
        self._unmute_presenter = unmute_presenter_mic or (lambda: None)
        self._on_speaking: Callable[[bool], None] = on_speaking_changed or (lambda _: None)

        self._client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        )
        self._stop_event = threading.Event()
        self._slide_history: list[str] = []
        self.busy = False  # True while reacting to a slide

        self._out_sr: int = GEM_SR
        self._out_channels: int = 1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._out_sr = self.audience_mini.media.get_output_audio_samplerate()
        self._out_channels = self.audience_mini.media.get_output_channels()
        self.audience_mini.media.start_playing()
        logger.info("[audience] started — out_sr=%d ch=%d", self._out_sr, self._out_channels)

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self.audience_mini.media.stop_playing()
        except Exception:
            pass
        logger.info("[audience] stopped")

    # ------------------------------------------------------------------
    # Entry point — called by HTTP server when presenter sends a slide event
    # ------------------------------------------------------------------

    def on_slide_presented(self, slide_number: int, script: str) -> None:
        """React to a slide being presented. Non-blocking."""
        logger.info("[audience] slide %d received (%d words)", slide_number, len(script.split()))
        threading.Thread(
            target=self._react_and_ask,
            args=(slide_number, script),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _react_and_ask(self, slide_number: int, script: str) -> None:
        if self._stop_event.is_set():
            return
        self.busy = True
        try:
            # 1. Immediate reaction — presenter is about to start the slide
            self._gesture_reaction(slide_number)

            # 2. Wait for presenter to finish speaking
            #    Estimate from word count so we don't cut them off
            word_count = len(script.split())
            estimated_duration = max(_MIN_WAIT, word_count / _WORDS_PER_SECOND)
            wait = estimated_duration + _POST_SPEECH_BUFFER
            logger.info("[audience] waiting %.1fs for presenter to finish (est. %.1fs for %d words)",
                        wait, estimated_duration, word_count)

            for _ in range(int(wait * 10)):
                if self._stop_event.is_set():
                    return
                time.sleep(0.1)

            if self._stop_event.is_set():
                return

            # 3. Generate a question
            question = self._generate_question(slide_number, script)
            if not question:
                logger.warning("[audience] no question generated for slide %d", slide_number)
                return

            logger.info("[audience] asking: %s", question)

            # 4. Question gesture, then speak
            self._mute_presenter()
            try:
                self._gesture_question()
                time.sleep(0.5)
                self._on_speaking(True)
                try:
                    self._speak(question)
                finally:
                    self._on_speaking(False)
            finally:
                time.sleep(0.5)
                self._unmute_presenter()

            self._slide_history.append(f"Slide {slide_number}: {script[:120]}")
        except Exception as e:
            logger.exception("[audience] _react_and_ask failed: %s", e)
        finally:
            self.busy = False

    def _gesture_reaction(self, slide_number: int) -> None:
        """Play a physical reaction gesture on the audience robot."""
        try:
            from reachy_mini.utils import create_head_pose
            if slide_number % 2 == 0:
                # Nod — pitch down then back
                self.audience_mini.goto_target(
                    head=create_head_pose(pitch=np.deg2rad(-12)), duration=0.4, method="minjerk"
                )
                self.audience_mini.goto_target(
                    head=create_head_pose(pitch=0.0), duration=0.4
                )
            else:
                # Excited — antennas up + slight head raise
                self.audience_mini.goto_target(
                    head=create_head_pose(pitch=np.deg2rad(8)),
                    antennas=np.deg2rad([40, 40]),
                    duration=0.5,
                    method="cartoon",
                )
                self.audience_mini.goto_target(
                    head=create_head_pose(pitch=0.0),
                    antennas=np.deg2rad([0.0, 0.0]),
                    duration=0.4,
                )
            logger.info("[audience] gesture done for slide %d", slide_number)
        except Exception as e:
            logger.warning("[audience] gesture failed: %s", e)

    def _gesture_question(self) -> None:
        """Play the 'asking a question' gesture — curious head tilt."""
        try:
            from reachy_mini.utils import create_head_pose
            self.audience_mini.goto_target(
                head=create_head_pose(roll=np.deg2rad(12)), duration=0.6, method="ease_in_out"
            )
            self.audience_mini.goto_target(
                head=create_head_pose(pitch=0.0), duration=0.5, method="ease_in_out"
            )
            logger.info("[audience] question gesture done")
        except Exception as e:
            logger.warning("[audience] question gesture failed: %s", e)

    def _generate_question(self, slide_number: int, script: str) -> str:
        history_ctx = "\n".join(self._slide_history[-3:]) if self._slide_history else "None"
        prompt = (
            f"The presenter just finished slide {slide_number} and said:\n\n"
            f"{script}\n\n"
            f"Previous slides context:\n{history_ctx}\n\n"
            "Ask one short, specific question about what was just said."
        )
        try:
            response = self._client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_AUDIENCE_SYSTEM_PROMPT,
                    temperature=0.8,
                    max_output_tokens=80,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = response.text.strip()
            # Sanity check — must be a complete, speakable sentence
            if len(text) < 8 or text.count('"') % 2 != 0:
                logger.warning("[audience] question looks malformed (%r) — skipping", text)
                return ""
            return text
        except Exception as e:
            logger.error("[audience] question generation failed: %s", e)
            return ""

    def _speak(self, text: str) -> None:
        try:
            audio_data = self._tts(text)
            if audio_data is None:
                return
            samples = decode_pcm(audio_data)
            # 9dB gain boost — Reachy Mini speaker needs amplification to be audible
            samples = np.clip(samples * 2.82, -1.0, 1.0).astype(np.float32)
            if self._out_sr != GEM_SR:
                n = int(len(samples) * self._out_sr / GEM_SR)
                samples = scipy_resample(samples, n).astype(np.float32)
            if self._out_channels > 1 and samples.ndim == 1:
                samples = np.stack([samples] * self._out_channels, axis=1)
            # Restart playing pipeline each time to avoid GStreamer idle timeout
            try:
                self.audience_mini.media.stop_playing()
            except Exception:
                pass
            self.audience_mini.media.start_playing()
            time.sleep(0.1)  # let pipeline initialise
            self.audience_mini.media.push_audio_sample(samples)
            duration = len(samples) / self._out_sr
            logger.info("[audience] speaking for %.1fs", duration)
            time.sleep(duration + 0.5)
        except Exception as e:
            logger.error("[audience] speak failed: %s", e)

    def _tts(self, text: str) -> Optional[bytes]:
        try:
            response = self._client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Zephyr",
                            )
                        )
                    ),
                ),
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    return part.inline_data.data
        except Exception as e:
            logger.error("[audience] TTS failed: %s", e)
        return None
