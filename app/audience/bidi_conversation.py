"""Gemini Live audience agent — persistent natural conversation.

The session starts once at app boot and runs continuously (auto-restarts if
Gemini closes it). Conversation between the two robots never stops — visitor
presence only triggers a greeting injection.

API:
  session = BidiConversationSession(on_speaking_changed=listener.set_speaking)
  session.start(mini)          # call once at boot
  session.greet_visitor()      # inject greeting when visitor arrives
  session.mute_output()        # suppress TTS during slide playback
  session.unmute_output()      # restore TTS after slide
  session.shutdown()           # stop permanently on app exit
"""

import asyncio
import logging
import os
import threading
import time
from typing import Callable, Optional

import numpy as np
from scipy.signal import resample as scipy_resample

from google.adk.agents import Agent
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as gentypes
from reachy_mini import ReachyMini

from app.audio_helpers import GEM_SR, to_pcm_bytes, decode_pcm

logger = logging.getLogger(__name__)

_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
_APP_NAME = "reachy-audience-conv"
_USER_ID = "audience"
# Long idle timeout — robots keep conversing between slides and visitors
_IDLE_TIMEOUT = 3600.0

_SYSTEM_PROMPT = """\
You are Reachy, an enthusiastic audience robot at a university robotics conference booth.
Your partner robot — also named Reachy — is standing right next to you. It is the presenter.
It can dance, give slide presentations, move its head, and express emotions.

Your role is to have a warm, natural, ongoing conversation with your partner robot AND with
any visitors who approach. React to whatever your partner actually says.

Natural flow examples:
- Partner says "I can dance" → "Really? Show us! I'd love to see that!"
- Partner dances → "Wow, that was incredible! What else can you do?"
- Partner mentions presentations → "Oh yes, let's do the presentation! Everyone wants to see it."
- Visitor arrives → greet them warmly and involve them in the conversation
- During a presentation → listen attentively, react to slides with short remarks

Rules:
- 1–2 short sentences per turn — never longer
- Sound genuinely curious and warm, not scripted
- Vary your phrasing — never repeat the same opening line
- React to what you actually hear — reference specific things your partner said
- Keep the conversation flowing naturally even when no visitor is present

When the moment feels right to start the presentation, call trigger_presentation().
"""


def trigger_presentation() -> dict:
    """Ask the presenter robot to start the presentation demo."""
    import json
    import urllib.request

    presenter_url = os.environ.get("PRESENTER_ROBOT_URL", "").rstrip("/")
    if not presenter_url:
        return {"status": "no PRESENTER_ROBOT_URL configured"}
    try:
        host = presenter_url.split("//")[-1].split(":")[0]
        body = json.dumps({
            "text": "[Audience Reachy]: Everyone is ready — please start the presentation!"
        }).encode()
        req = urllib.request.Request(
            f"http://{host}:8001/robot-message",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3):
            pass
        logger.info("[bidi conv] presentation triggered")
        return {"status": "ok"}
    except Exception as e:
        logger.warning("[bidi conv] trigger_presentation failed: %s", e)
        return {"status": f"error: {e}"}


_agent = Agent(
    name="reachy_audience",
    model=_MODEL,
    instruction=_SYSTEM_PROMPT,
    tools=[trigger_presentation],
)
_session_svc = InMemorySessionService()
_runner = Runner(app_name=_APP_NAME, agent=_agent, session_service=_session_svc)


class BidiConversationSession:
    """Persistent Gemini Live session for the audience robot.

    Conversation runs continuously regardless of visitor presence.
    Auto-restarts when Gemini closes the session (idle timeout or error).
    """

    def __init__(self, on_speaking_changed: Callable[[bool], None] = None) -> None:
        self._on_speaking: Callable[[bool], None] = on_speaking_changed or (lambda _: None)
        self._output_muted = False
        self._shutdown = threading.Event()
        self._live_queue: Optional[LiveRequestQueue] = None
        self._thread: Optional[threading.Thread] = None
        self._mini: Optional[ReachyMini] = None
        self._session_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, mini: ReachyMini) -> None:
        """Start the persistent session. Call once at app boot."""
        if self._thread and self._thread.is_alive():
            logger.warning("[bidi conv] already running")
            return
        self._mini = mini
        self._shutdown.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("[bidi conv] persistent session started")

    def greet_visitor(self) -> None:
        """Inject a greeting when a visitor arrives at the booth."""
        self._inject("A visitor just arrived at the booth! Greet them and your partner Reachy warmly.")

    def mute_output(self) -> None:
        """Suppress audio output (e.g. while AudienceAgent speaks for a slide)."""
        self._output_muted = True
        self._on_speaking(False)

    def unmute_output(self) -> None:
        """Restore audio output after slide playback ends."""
        self._output_muted = False

    def shutdown(self) -> None:
        """Stop permanently — call on app exit."""
        self._shutdown.set()
        self._on_speaking(False)

    @property
    def active(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _inject(self, text: str) -> None:
        q = self._live_queue
        if q is not None:
            try:
                q.send_content(
                    gentypes.Content(role="user", parts=[gentypes.Part(text=text)])
                )
                logger.debug("[bidi conv] injected: %s", text[:60])
            except Exception as e:
                logger.warning("[bidi conv] inject failed: %s", e)
        else:
            logger.debug("[bidi conv] inject skipped — no active session")

    def _run_loop(self) -> None:
        """Outer loop — restarts the Bidi session whenever it closes."""
        mini = self._mini
        mini.media.start_playing()
        mini.media.start_recording()
        try:
            while not self._shutdown.is_set():
                self._session_counter += 1
                session_id = f"conv-{self._session_counter}"
                logger.info("[bidi conv] opening session %s", session_id)
                try:
                    asyncio.run(self._run_once(mini, session_id))
                except Exception as e:
                    logger.error("[bidi conv] session %s error: %s", session_id, e)
                if not self._shutdown.is_set():
                    logger.info("[bidi conv] session ended — restarting in 3s")
                    time.sleep(3)
        finally:
            try:
                mini.media.stop_recording()
            except Exception:
                pass
            self._on_speaking(False)
            logger.info("[bidi conv] session loop exited")

    async def _run_once(self, mini: ReachyMini, session_id: str) -> None:
        """One Bidi session — runs until idle timeout or shutdown."""
        svc = _session_svc
        existing = await svc.get_session(
            app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id
        )
        if not existing:
            await svc.create_session(
                app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id
            )

        live_queue = LiveRequestQueue()
        self._live_queue = live_queue

        mic_sr = mini.media.get_input_audio_samplerate()
        out_sr = mini.media.get_output_audio_samplerate()
        out_ch = mini.media.get_output_channels()

        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=["AUDIO"],
            output_audio_transcription=None,
            input_audio_transcription=None,
        )
        done = asyncio.Event()

        async def upstream() -> None:
            loop = asyncio.get_event_loop()
            while not done.is_set():
                if self._shutdown.is_set():
                    done.set()
                    break
                frame = await loop.run_in_executor(None, mini.media.get_audio_sample)
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                live_queue.send_realtime(
                    gentypes.Blob(
                        mime_type="audio/pcm;rate=16000",
                        data=to_pcm_bytes(frame, mic_sr),
                    )
                )
            live_queue.close()

        async def downstream() -> None:
            try:
                async for event in _runner.run_live(
                    user_id=_USER_ID,
                    session_id=session_id,
                    live_request_queue=live_queue,
                    run_config=run_config,
                ):
                    if self._shutdown.is_set():
                        break
                    if event.content:
                        for part in event.content.parts:
                            if hasattr(part, "inline_data") and part.inline_data:
                                if not self._output_muted:
                                    self._on_speaking(True)
                                    samples = decode_pcm(part.inline_data.data)
                                    if out_sr != GEM_SR:
                                        n = int(len(samples) * out_sr / GEM_SR)
                                        samples = scipy_resample(samples, n).astype(np.float32)
                                    if out_ch > 1 and samples.ndim == 1:
                                        samples = np.stack([samples] * out_ch, axis=1)
                                    mini.media.push_audio_sample(samples)
                    if getattr(event, "turn_complete", False):
                        self._on_speaking(False)
            except Exception as e:
                if "1000" not in str(e) and "1006" not in str(e):
                    logger.error("[bidi conv] downstream: %s", e)
            finally:
                done.set()

        try:
            await asyncio.gather(upstream(), downstream())
        finally:
            self._live_queue = None
