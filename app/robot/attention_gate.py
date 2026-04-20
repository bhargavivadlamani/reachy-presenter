"""Attention-gated audio using the sd_attention SDK.

When sd_attention is installed:
  - AttentionProcessor classifies speech every 250ms
  - Classes: 0 = SILENT, 1 = TO_HUMAN, 2 = TO_COMPUTER (Reachy)
  - Only audio classified as TO_COMPUTER is forwarded to Gemini
  - mic_muted + is_responding flags prevent Reachy from hearing itself

When sd_attention is NOT installed (or fails to init):
  - Falls back transparently to always-on streaming (original behavior)
  - No code changes needed in callers — gate.available == False
    means the caller should use mini.media.get_audio_sample() directly

Integration design
------------------
The SDK's on_speech_audio_ready callback delivers complete,
attention-filtered utterances as int16 numpy arrays at 16kHz.
We chunk these into 512-sample pieces and queue them so that
_run_bidi_async's upstream() loop can consume them identically
to how it consumes raw mic frames — no changes to the streaming loop itself.

Audio recording is handled exclusively by ReachyMiniManager when the gate
is active, so run_for_robot() must NOT call mini.media.start_recording()
in that case (check gate.available before starting recording).

Note on media backends
----------------------
sd_attention uses media_backend='SOUNDDEVICE_OPENCV' (sounddevice + OpenCV).
The presenter app normally uses the default GStreamer backend for recording.
When the gate is active, recording is handled by ReachyMiniManager, and
mini.media is only used for audio OUTPUT (start_playing / push_audio_sample).
"""

import logging
import queue
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 512     # samples per chunk fed into Gemini's upstream loop
_SDK_TIMEOUT = 10.0   # seconds to wait for ReachyMiniManager init


class AttentionGate:
    """Gates audio to Gemini based on sd_attention speech classification.

    Typical usage in run_for_robot():

        gate = AttentionGate()
        gate.setup(mini)          # tries sd_attention; graceful no-op if missing

        if gate.available:
            # SDK handles recording — do NOT call mini.media.start_recording()
            mini.media.start_playing()
        else:
            mini.media.start_playing()
            mini.media.start_recording()

        def _get_frame():
            if gate.available:
                return gate.next_frame()   # attention-filtered chunks
            return mini.media.get_audio_sample()  # always-on

        def _on_audio_start():
            gate.set_responding(True)    # Reachy speaking → mute SDK mic

        def _unmute():
            gate.set_responding(False)   # Reachy done → SDK may listen again
    """

    def __init__(self) -> None:
        self._available = False
        self._processor = None
        self._reachy_mgr = None
        self._audio_queue: queue.Queue = queue.Queue(maxsize=400)

    # ------------------------------------------------------------------
    # Setup — call once before run_for_robot
    # ------------------------------------------------------------------

    def setup(self, mini) -> bool:
        """Try to initialise sd_attention.  Returns True if active."""
        try:
            from sd_attention import AttentionProcessor, AttentionConfig
            from sd_attention.reachy_manager import ReachyMiniManager
        except ImportError:
            logger.info(
                "AttentionGate: sd_attention not installed — always-on mode.\n"
                "  To enable attention filtering: pip install sd_attention-*.whl"
            )
            return False

        try:
            reachy_mgr = ReachyMiniManager.get_instance()
            ok = reachy_mgr.initialize(
                localhost_only=True,
                spawn_daemon=False,
                use_sim=False,
                timeout=_SDK_TIMEOUT,
                media_backend='SOUNDDEVICE_OPENCV',
            )
            if not ok:
                logger.warning("AttentionGate: ReachyMiniManager init failed — always-on mode.")
                return False

            reachy_mgr.start_audio_recording()
            self._reachy_mgr = reachy_mgr

            config = AttentionConfig(
                auto_collect=True,
                class_2_threshold=3,
                class_0_threshold=3,
                min_speech_duration=1.0,
                llm_buffer_max_duration=30.0,
                vad_threshold=0.4,
                max_faces=3,
            )

            self._processor = AttentionProcessor(
                reachy=reachy_mgr,
                config=config,
                logger=None,
            )

            self._register_callbacks()
            self._processor.start()
            self._available = True
            logger.info(
                "AttentionGate: sd_attention active — Reachy will only respond "
                "when spoken to directly (class TO_COMPUTER)."
            )
            return True

        except Exception as e:
            logger.warning(f"AttentionGate: init error ({e}) — always-on mode.")
            return False

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _register_callbacks(self) -> None:
        @self._processor.on_listening_start
        def _on_start():
            logger.info("[Attention] Listening — user is addressing the robot")

        @self._processor.on_speech_audio_ready
        def _on_speech(audio_pcm16: np.ndarray, duration_sec: float):
            """Complete speech utterance directed at the robot.

            audio_pcm16: int16 numpy array, mono, 16kHz — already Gemini's format.
            """
            logger.info(f"[Attention] Speech ready ({duration_sec:.1f}s) — queuing for Gemini")
            # Convert int16 → float32 (same representation as mini.media.get_audio_sample())
            audio_f32 = audio_pcm16.astype(np.float32) / 32768.0
            # Chunk into bite-sized pieces the upstream loop can consume
            for i in range(0, len(audio_f32), _CHUNK_SIZE):
                chunk = audio_f32[i : i + _CHUNK_SIZE]
                try:
                    self._audio_queue.put_nowait(chunk)
                except queue.Full:
                    logger.debug("[Attention] Audio queue full — dropping chunk")

        @self._processor.on_listening_cancelled
        def _on_cancelled():
            logger.info("[Attention] Cancelled — speech too short or not directed at robot")

        @self._processor.on_prediction
        def _on_prediction(prediction: int, confidence: float, features):
            labels = {0: "SILENT", 1: "TO_HUMAN", 2: "TO_COMPUTER"}
            label = labels.get(prediction, f"?({prediction})")
            logger.debug(f"[Attention] {label} {confidence*100:.0f}%")

    # ------------------------------------------------------------------
    # Public API used by run_for_robot
    # ------------------------------------------------------------------

    def next_frame(self) -> Optional[np.ndarray]:
        """Return the next attention-filtered audio frame, or None if empty.

        Call this in place of mini.media.get_audio_sample() when gate.available.
        """
        try:
            return self._audio_queue.get_nowait()
        except queue.Empty:
            return None

    def set_responding(self, responding: bool) -> None:
        """Notify the SDK whether Reachy is currently speaking.

        When True  — SDK mutes its own mic collection so Reachy doesn't
                     hear herself and loop.
        When False — SDK resumes listening for the next utterance.
        """
        if self._processor is not None:
            self._processor.is_responding = responding
            self._processor.mic_muted = responding

    @property
    def available(self) -> bool:
        """True if sd_attention is active and filtering audio."""
        return self._available

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def stop(self) -> None:
        if self._processor is not None:
            try:
                self._processor.stop()
            except Exception:
                pass
        if self._reachy_mgr is not None:
            try:
                self._reachy_mgr.shutdown_cleanup()
            except Exception:
                pass
