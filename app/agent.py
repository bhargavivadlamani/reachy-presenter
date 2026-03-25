"""ADK-based agent using Gemini Bidi streaming.

CLI mode (text in / text out, no robot required):
    python -m app.agent

Simulator / audio mode (sounddevice audio, simulator for robot control):
    python -m app.agent --audio

Real robot mode:
    from app.agent import run_for_robot
    run_for_robot(mini, script, document_text)
"""

import asyncio
import logging
import queue as _queue
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from reachy_mini import ReachyMini
from scipy.signal import resample as scipy_resample  # used in run_for_robot resampling
from app.audio_helpers import MIC_SR, GEM_SR, VAD_THRESHOLD, to_pcm_bytes, decode_pcm
from app.tools.present_slide import present_slide, set_mini
from app.tools.load_presentation import load_presentation

load_dotenv()

# ADK logs WebSocket close errors before we handle them — suppress the noise
for _noisy_logger in (
    "google.adk.flows.llm_flows.base_llm_flow",
    "google_adk.google.adk.flows.llm_flows.base_llm_flow",
):
    logging.getLogger(_noisy_logger).setLevel(logging.CRITICAL)
logging.getLogger("google.adk.models.google_llm").setLevel(logging.DEBUG)
logging.getLogger("google_adk.google.adk.models.google_llm").setLevel(logging.DEBUG)

_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
_APP_NAME = "reachy-presenter"
_USER_ID = "presenter"
_SESSION_ID = "main"
_PRESENTATION_PATH = Path("/home/pollen/ReachyMini1.pdf")

_SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text().strip()
if _PRESENTATION_PATH.exists():
    _SYSTEM_PROMPT += f"\n\nDefault presentation file: `{_PRESENTATION_PATH}`. When the user asks to start, give, or begin the presentation, call load_presentation() with this path."


# ---------------------------------------------------------------------------
# Application singletons (created once at import time)
# ---------------------------------------------------------------------------

_agent = Agent(
    name="reachy_presenter",
    model=_MODEL,
    instruction=_SYSTEM_PROMPT,
    tools=[present_slide, load_presentation],
)

_session_service = InMemorySessionService()
_runner = Runner(
    app_name=_APP_NAME,
    agent=_agent,
    session_service=_session_service,
)


async def _get_or_create_session(user_id: str, session_id: str) -> None:
    session = await _session_service.get_session(
        app_name=_APP_NAME, user_id=user_id, session_id=session_id
    )
    if not session:
        await _session_service.create_session(
            app_name=_APP_NAME, user_id=user_id, session_id=session_id
        )



# ---------------------------------------------------------------------------
# Core streaming loop — audio I/O is injected via callbacks
# ---------------------------------------------------------------------------

async def _run_bidi_async(
    get_audio_frame: Callable[[], Optional[np.ndarray]],
    push_audio: Callable[[np.ndarray], None],
    mic_sr: int,
    clear_audio: Callable[[], None] = lambda: None,
    unmute_audio: Callable[[], None] = lambda: None,
    initial_text: str = "",
    idle_timeout: float = 30.0,
) -> None:
    await _get_or_create_session(_USER_ID, _SESSION_ID)
    live_queue = LiveRequestQueue()
    run_config = RunConfig(
        streaming_mode=StreamingMode.BIDI,
        response_modalities=["AUDIO"],
        output_audio_transcription=None,
        input_audio_transcription=None,
    )

    done = asyncio.Event()
    idle_task: asyncio.Task | None = None
    _last_idle_restart: float = 0.0

    def _restart_idle_timer() -> None:
        nonlocal idle_task
        if idle_task and not idle_task.done():
            idle_task.cancel()
        async def _idle() -> None:
            await asyncio.sleep(idle_timeout)
            print(f"[idle] No activity for {idle_timeout}s — closing session.")
            done.set()
        idle_task = asyncio.create_task(_idle())

    async def upstream() -> None:
        nonlocal _last_idle_restart
        if initial_text:
            live_queue.send_content(
                types.Content(role="user", parts=[types.Part(text=initial_text)])
            )
        loop = asyncio.get_event_loop()
        while not done.is_set():
            frame = await loop.run_in_executor(None, get_audio_frame)
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            # Throttle timer restarts to once/sec — high-fps sources (e.g. 77fps
            # robot mic) would otherwise thrash asyncio task creation/cancellation.
            now = loop.time()
            if now - _last_idle_restart >= 1.0:
                _last_idle_restart = now
                _restart_idle_timer()
            pcm = to_pcm_bytes(frame, mic_sr)
            live_queue.send_realtime(
                types.Blob(mime_type="audio/pcm;rate=16000", data=pcm)
            )
        live_queue.close()

    async def downstream() -> None:
        try:
            async for event in _runner.run_live(
                user_id=_USER_ID,
                session_id=_SESSION_ID,
                live_request_queue=live_queue,
                run_config=run_config,
            ):
                if getattr(event, "interrupted", False):
                    clear_audio()
                if event.content:
                    for part in event.content.parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            samples = decode_pcm(part.inline_data.data)
                            push_audio(samples)
                if getattr(event, "turn_complete", False):
                    unmute_audio()
                    _restart_idle_timer()
        except Exception as e:
            msg = str(e)
            if "1000" not in msg and "1006" not in msg:
                raise
        finally:
            done.set()

    try:
        await asyncio.gather(upstream(), downstream())
    finally:
        if idle_task and not idle_task.done():
            idle_task.cancel()


# ---------------------------------------------------------------------------
# Real robot mode — audio via mini.media (GStreamer on the robot hardware)
# ---------------------------------------------------------------------------

def run_for_robot(mini: ReachyMini) -> None:
    """Start a bidi agent session on the real robot, managing media lifecycle."""
    robot_mic_sr = mini.media.get_input_audio_samplerate()
    robot_out_sr = mini.media.get_output_audio_samplerate()

    def _get_frame() -> Optional[np.ndarray]:
        return mini.media.get_audio_sample()

    robot_out_channels = mini.media.get_output_channels()

    def _push(samples: np.ndarray) -> None:
        if robot_out_sr != GEM_SR:
            n = int(len(samples) * robot_out_sr / GEM_SR)
            samples = scipy_resample(samples, n).astype(np.float32)
        if robot_out_channels > 1 and samples.ndim == 1:
            samples = np.stack([samples] * robot_out_channels, axis=1)
        mini.media.push_audio_sample(samples)

    mini.media.start_playing()
    mini.media.start_recording()
    try:
        asyncio.run(_run_bidi_async(
            get_audio_frame=_get_frame,
            push_audio=_push,
            mic_sr=robot_mic_sr,
        ))
    finally:
        mini.media.stop_recording()
        mini.media.stop_playing()


# ---------------------------------------------------------------------------
# Simulator / sounddevice mode — audio via PC mic + speakers
# ---------------------------------------------------------------------------

def run_audio_conversation(initial_text: str = "") -> None:
    """Start a bidi conversation using sounddevice for audio I/O."""
    mic_q: _queue.SimpleQueue = _queue.SimpleQueue()

    # Output state — accessed from both sounddevice callback thread and async loop
    _out_chunks: list = []
    _out_lock = threading.Lock()
    _out_state = {
        "remainder": np.array([], dtype=np.float32),
        "muted": False,
        "interrupted": False,  # local VAD flagged interruption, waiting for turn_complete
    }

    def _out_callback(outdata: np.ndarray, frames: int, time, status) -> None:
        outdata[:] = 0
        if _out_state["muted"]:
            return
        needed, pos = frames, 0
        with _out_lock:
            buf = _out_state["remainder"]
            while needed > 0:
                if len(buf) == 0:
                    if not _out_chunks:
                        break
                    buf = _out_chunks.pop(0)
                n = min(len(buf), needed)
                outdata[pos:pos + n, 0] = buf[:n]
                buf = buf[n:]
                pos += n
                needed -= n
            _out_state["remainder"] = buf

    def _get_frame() -> Optional[np.ndarray]:
        try:
            return mic_q.get_nowait()
        except _queue.Empty:
            return None

    def _push(samples: np.ndarray) -> None:
        with _out_lock:
            _out_chunks.append(samples)

    def _flush_and_unmute() -> None:
        """Clear output queue, flush hardware buffer, and unmute."""
        _out_state["muted"] = True
        with _out_lock:
            _out_chunks.clear()
            _out_state["remainder"] = np.array([], dtype=np.float32)
        out_stream.abort()
        out_stream.start()
        _out_state["interrupted"] = False
        _out_state["muted"] = False

    def _clear() -> None:
        """Called when server confirms interruption — same as flush."""
        _flush_and_unmute()

    def _unmute() -> None:
        """Called on turn_complete. If local VAD flagged interruption, flush first."""
        if _out_state["interrupted"]:
            _flush_and_unmute()
        else:
            _out_state["muted"] = False

    def _mic_callback(indata: np.ndarray, frames: int, time, status) -> None:
        frame = indata[:, 0].copy()
        mic_q.put(frame)
        # Local VAD: only act when bot is actively playing (ignore ambient noise)
        with _out_lock:
            bot_playing = len(_out_chunks) > 0 or len(_out_state["remainder"]) > 0
        if bot_playing and not _out_state["interrupted"] and \
                np.sqrt(np.mean(frame ** 2)) > VAD_THRESHOLD:
            _out_state["muted"] = True
            _out_state["interrupted"] = True

    mic_stream = sd.InputStream(
        samplerate=MIC_SR, channels=1, dtype="float32",
        callback=_mic_callback,
        blocksize=512,
        latency="low",
    )
    out_stream = sd.OutputStream(
        samplerate=GEM_SR, channels=1, dtype="float32",
        callback=_out_callback,
        blocksize=512,
        latency="low",
    )

    with mic_stream, out_stream:
        asyncio.run(_run_bidi_async(
            get_audio_frame=_get_frame,
            push_audio=_push,
            mic_sr=MIC_SR,
            clear_audio=_clear,
            unmute_audio=_unmute,
            initial_text=initial_text,
        ))


# ---------------------------------------------------------------------------
# CLI mode: text in / text out via run_async (no robot, no LiveRequestQueue)
# ---------------------------------------------------------------------------

async def _run_cli_async() -> None:
    await _get_or_create_session("cli-user", "cli-session")

    while True:
        try:
            text = input("You: ").strip()
        except EOFError:
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            break

        content = types.Content(role="user", parts=[types.Part(text=text)])
        print("Agent: ", end="", flush=True)
        async for event in _runner.run_async(
            user_id="cli-user", session_id="cli-session", new_message=content
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        print(part.text, end="", flush=True)
        print()


if __name__ == "__main__":
    import sys
    if "--robot" in sys.argv:
        mini = ReachyMini()
        set_mini(mini)
        print("Connected to robot. Just start talking.\n")
        with mini:
            run_for_robot(mini)
    elif "--audio" in sys.argv:
        set_mini(ReachyMini(
            connection_mode="localhost_only",
            timeout=5.0,
            media_backend="no_media",
        ))
        print("Connected to simulator. Just start talking.\n")
        run_audio_conversation()
    else:
        print("Reachy Agent (text mode). Type 'quit' to exit.\n")
        asyncio.run(_run_cli_async())
