"""Audience agent — second Reachy Mini that reacts and asks questions.

Architecture:
- Physical reactions via DualRobotManager (nod, applaud, gasp, etc.)
- Question generation via Gemini Flash text (lightweight, not Bidi)
- Speech output via Gemini TTS → push_audio_sample on audience robot
- Mic muting: presenter mutes its own mic while audience speaks, and vice versa

This module is fully standalone — it does not import or modify anything
from app/agent.py or the presenter tools.
"""

import logging
import threading
import time
import asyncio
import os
import numpy as np
from typing import Optional, Callable

from google import genai
from google.genai import types
from scipy.signal import resample as scipy_resample
from reachy_mini import ReachyMini

from reachy_mini_conversation_app.orchestrator.dual_robot_manager import (
    DualRobotManager,
    AudienceReaction,
)

from app.audio_helpers import GEM_SR, decode_pcm

logger = logging.getLogger(__name__)

_AUDIENCE_SYSTEM_PROMPT = """You are a curious, engaged audience member watching a robot give a presentation.
You ask short, natural questions (1-2 sentences max) about what was just presented.
Be genuinely curious — ask about specifics, implications, or things you didn't fully understand.
Never repeat the same question. Never ask generic questions like "can you tell me more?".
Respond with ONLY the question text, nothing else."""


class AudienceAgent:
    """Controls the audience Reachy Mini — reactions + voice questions.

    Usage:
        audience = AudienceAgent(audience_mini, presenter_mini)
        audience.start()
        # When a slide is done:
        audience.on_slide_presented(slide_number, script_text)
        # On shutdown:
        audience.stop()
    """

    def __init__(
        self,
        audience_mini: ReachyMini,
        presenter_mini: ReachyMini,
        mute_presenter_mic: Optional[Callable] = None,
        unmute_presenter_mic: Optional[Callable] = None,
    ) -> None:
        self.audience_mini = audience_mini
        self._dual = DualRobotManager(
            presenter_robot=presenter_mini,
            presenter_movement=None,   # we don't need MovementManager — not used in reaction path
            audience_robot=audience_mini,
            audience_movement=None,
        )
        self._mute_presenter = mute_presenter_mic or (lambda: None)
        self._unmute_presenter = unmute_presenter_mic or (lambda: None)

        self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        self._stop_event = threading.Event()
        self._slide_history: list[str] = []

        # Robot audio specs (populated on start)
        self._out_sr: int = GEM_SR
        self._out_channels: int = 1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start media on audience robot."""
        self._out_sr = self.audience_mini.media.get_output_audio_samplerate()
        self._out_channels = self.audience_mini.media.get_output_channels()
        self.audience_mini.media.start_playing()
        logger.info("[audience] started, out_sr=%d ch=%d", self._out_sr, self._out_channels)

    def stop(self) -> None:
        """Stop audience agent cleanly."""
        self._stop_event.set()
        try:
            self.audience_mini.media.stop_playing()
        except Exception:
            pass
        logger.info("[audience] stopped")

    # ------------------------------------------------------------------
    # Called by present_slide tool after each slide
    # ------------------------------------------------------------------

    def on_slide_presented(self, slide_number: int, script: str) -> None:
        """React to a completed slide and ask a question. Non-blocking."""
        threading.Thread(
            target=self._react_and_ask,
            args=(slide_number, script),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _react_and_ask(self, slide_number: int, script: str) -> None:
        """Physical reaction → short pause → generate question → speak it."""
        if self._stop_event.is_set():
            return

        # 1. Physical reaction based on slide number
        if slide_number == 1:
            self._dual.on_presentation_started()
        else:
            self._dual.on_slide_ended(slide_number - 1)
            self._dual.on_slide_started(slide_number)
        self._dual.update()

        # 2. Wait for presenter to finish speaking (simple fixed delay)
        time.sleep(2.5)

        if self._stop_event.is_set():
            return

        # 3. Generate a question
        question = self._generate_question(slide_number, script)
        if not question:
            logger.warning("[audience] no question generated for slide %d", slide_number)
            return

        logger.info("[audience] question: %s", question)

        # 4. Mute presenter mic, do question gesture, speak
        self._mute_presenter()
        try:
            self._dual.ask_question_gesture()
            time.sleep(0.4)   # let gesture start
            self._speak(question)
        finally:
            time.sleep(0.5)   # small gap after speaking before presenter mic opens
            self._unmute_presenter()

        # 5. Track history so questions don't repeat
        self._slide_history.append(f"Slide {slide_number}: {script[:120]}")

    def _generate_question(self, slide_number: int, script: str) -> str:
        """Call Gemini Flash to generate a short audience question."""
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
                ),
            )
            return response.text.strip()
        except Exception as e:
            logger.error("[audience] question generation failed: %s", e)
            return ""

    def _speak(self, text: str) -> None:
        """Convert text to speech and push audio to audience robot."""
        try:
            audio_data = self._tts(text)
            if audio_data is None:
                return
            # Resample if robot SR differs from Gemini's 24kHz output
            samples = decode_pcm(audio_data)
            if self._out_sr != GEM_SR:
                n = int(len(samples) * self._out_sr / GEM_SR)
                samples = scipy_resample(samples, n).astype(np.float32)
            if self._out_channels > 1 and samples.ndim == 1:
                samples = np.stack([samples] * self._out_channels, axis=1)
            self.audience_mini.media.push_audio_sample(samples)
            # Wait for audio to finish playing
            duration = len(samples) / self._out_sr
            time.sleep(duration + 0.3)
        except Exception as e:
            logger.error("[audience] speak failed: %s", e)

    def _tts(self, text: str) -> Optional[bytes]:
        """Use Gemini to synthesise speech, return raw 24kHz int16 PCM bytes."""
        try:
            response = self._client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Zephyr",   # different voice from presenter
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
