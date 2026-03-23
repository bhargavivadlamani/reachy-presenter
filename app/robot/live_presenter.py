"""Gemini Live API integration: streaming TTS + audience Q&A in one session.

The Live API handles everything:
  - Speaks the slide script (TTS)
  - Listens to the mic with built-in VAD + echo cancellation
  - Detects audience interruptions natively (sc.interrupted)
  - Answers questions, then waits for "continue" / "okay" to advance

The presenter says "continue" or "okay" (or similar) → Gemini calls the
advance_slide() function tool → done is set → next slide begins.
"""

import asyncio
import contextlib
import os
import threading
import time

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from scipy.signal import resample as scipy_resample

load_dotenv()

_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
_MIC_SR = 16000   # Gemini expects 16kHz PCM mic input
_GEM_SR = 24000   # Gemini outputs 24kHz 16-bit PCM
_MIC_GAIN = 2     # raw speech ~0.09–0.12 → ×2 = 0.18–0.25, good for VAD without clipping

_client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
    http_options={"api_version": "v1beta"},
)

_ADVANCE_TOOL = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="advance_slide",
        description=(
            "Call this function when the presenter or audience says 'continue', "
            "'next', 'okay', 'let's move on', 'go ahead', or any phrase that "
            "clearly signals they are ready to proceed to the next slide."
        ),
    )
])


def present_slide(mini, script: str, document_text: str) -> None:
    """Speak a slide, handle audience Q&A, then return. Blocks until done."""
    asyncio.run(_run(mini, script, document_text))


async def _run(mini, script: str, document_text: str) -> None:
    robot_sr = mini.media.get_output_audio_samplerate()
    robot_mic_sr = mini.media.get_input_audio_samplerate()
    print(f"[audio] output={robot_sr}Hz  mic={robot_mic_sr}Hz")
    loop = asyncio.get_running_loop()
    done = asyncio.Event()
    mic_enabled = asyncio.Event()  # only send mic audio after TTS is done (prevents echo)

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=(
            "You are a presenter delivering a live talk. "
            "When given a script, read it OUT LOUD word for word, exactly as written. "
            "Do not summarize, shorten, or skip any part of the script. Speak every sentence. "
            "After you have finished reading the full script, ask: 'Does anyone have any questions? Or say continue to move on to the next slide.' "
            "If someone asks a question, answer it concisely in 2-3 sentences using the presentation context below, "
            "then ask if there are any more questions. "
            "When the presenter or audience says 'continue', 'next', 'no questions', 'okay', or any phrase "
            "indicating they are ready to move on, call the advance_slide() function.\n\n"
            "Presentation context:\n" + document_text
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
            )
        ),
        tools=[_ADVANCE_TOOL],
    )

    # Face tracking runs for the entire slide in its own thread
    stop_tracking = threading.Event()
    from app.robot.vision import track_faces_during_speech
    face_thread = threading.Thread(
        target=track_faces_during_speech, args=(mini, stop_tracking), daemon=True
    )
    face_thread.start()

    try:
        async with _client.aio.live.connect(model=_MODEL, config=config) as session:
            # send_client_content with turn_complete=True tells Gemini to speak the
            # script immediately regardless of VAD mode (manual activity detection).
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text=script)],
                ),
                turn_complete=True,
            )

            # Mic → queue: robot SDK (same pipeline as speaker → hardware AEC works).
            # start_recording() must already be called in main.py before entering here.
            mic_queue: asyncio.Queue[bytes] = asyncio.Queue()

            def mic_reader():
                logged_first = False
                while not done.is_set():
                    frame = mini.media.get_audio_sample()
                    if frame is None:
                        time.sleep(0.01)   # no data yet — don't busy-spin
                        continue
                    # SDK always returns (num_samples, channels) — average to mono
                    if frame.ndim == 2:
                        frame = frame.mean(axis=1).astype(np.float32)
                    # Resample to 16kHz if device runs at a different rate (e.g. 44100)
                    if robot_mic_sr != _MIC_SR:
                        n = int(len(frame) * _MIC_SR / robot_mic_sr)
                        frame = scipy_resample(frame, n).astype(np.float32)
                    if not logged_first:
                        logged_first = True
                        print(f"[mic_reader] First frame: {len(frame)} samples, native={robot_mic_sr}Hz, max={abs(frame).max():.4f}")
                    # float32 → int16 PCM (Gemini expects 16-bit signed PCM at 16kHz)
                    pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
                    try:
                        loop.call_soon_threadsafe(mic_queue.put_nowait, pcm.tobytes())
                    except RuntimeError:
                        break  # event loop closed

            threading.Thread(target=mic_reader, daemon=True).start()

            # Idle timeout: auto-advance if nobody says "continue" within 30s
            idle_task: asyncio.Task | None = None

            def restart_idle_timer():
                nonlocal idle_task
                if idle_task and not idle_task.done():
                    idle_task.cancel()
                async def _idle():
                    await asyncio.sleep(30)
                    print("[idle timeout] No response — advancing to next slide.")
                    done.set()
                idle_task = asyncio.create_task(_idle())

            async def send_mic():
                """Client-side VAD: buffer speech, send as send_client_content() on silence."""
                THRESH     = 0.05   # normalised amplitude threshold
                START_HOLD = 8      # loud frames to confirm speech start (~130ms)
                END_HOLD   = 40     # quiet frames to confirm silence (~640ms)

                speaking   = False
                loud_n     = 0
                quiet_n    = 0
                audio_buf: list[bytes] = []

                while not done.is_set():
                    try:
                        data = await asyncio.wait_for(mic_queue.get(), timeout=0.5)
                        if not mic_enabled.is_set():
                            continue  # discard while TTS is playing

                        # Amplify
                        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                        samples = np.clip(samples * _MIC_GAIN, -32767, 32767).astype(np.int16)
                        amp = float(np.abs(samples).max()) / 32767.0
                        data = samples.tobytes()

                        if amp > THRESH:
                            loud_n += 1
                            quiet_n = 0
                            if not speaking and loud_n >= START_HOLD:
                                speaking  = True
                                loud_n    = 0
                                audio_buf = []
                                print(f"[vad] Speech detected (amp={amp:.3f})")
                        else:
                            quiet_n += 1
                            loud_n   = 0
                            if speaking and quiet_n >= END_HOLD:
                                speaking = False
                                quiet_n  = 0
                                if audio_buf:
                                    pcm = b"".join(audio_buf)
                                    print(f"[vad] Silence — sending {len(pcm)//512}×256-sample chunks")
                                    # Same channel as the initial script → Gemini responds
                                    await session.send_client_content(
                                        turns=types.Content(
                                            role="user",
                                            parts=[types.Part(
                                                inline_data=types.Blob(
                                                    data=pcm,
                                                    mime_type="audio/pcm;rate=16000",
                                                )
                                            )],
                                        ),
                                        turn_complete=True,
                                    )
                                audio_buf = []

                        if speaking:
                            audio_buf.append(data)

                    except asyncio.TimeoutError:
                        pass
                    except Exception as e:
                        print(f"[send_mic error] {e}")
                        done.set()
                        break

            async def receive_audio():
                """Play Gemini's audio, handle interruptions, advance on function call."""
                audio_start: float | None = None
                audio_pushed_s: float = 0.0
                audio_chunk_count = 0
                listening = False

                async for response in session.receive():
                    # Diagnostic: log all non-audio server events
                    if not response.data:
                        print(f"[recv] {response}")

                    # --- Audio playback ---
                    if response.data:
                        samples = (
                            np.frombuffer(response.data, dtype=np.int16)
                            .astype(np.float32) / 32768.0
                        )
                        if robot_sr != _GEM_SR:
                            n = int(len(samples) * robot_sr / _GEM_SR)
                            samples = scipy_resample(samples, n).astype(np.float32)
                        if audio_start is None:
                            audio_start = time.monotonic()
                            if listening:
                                print("[audio] Gemini answering...")
                                mic_enabled.clear()  # mute mic while Gemini answers (prevent echo)
                            else:
                                print("[audio] Gemini started speaking")
                        audio_pushed_s += len(samples) / robot_sr
                        audio_chunk_count += 1
                        mini.media.push_audio_sample(samples)

                    sc = response.server_content
                    if sc:
                        if sc.interrupted:
                            print(f"[interrupted] after {audio_chunk_count} chunks — echo or user spoke")
                            audio_start = None
                            audio_pushed_s = 0.0
                            audio_chunk_count = 0

                        if sc.turn_complete:
                            # Save before reset so we can wait for speaker buffer to drain
                            _started = audio_start
                            _pushed_s = audio_pushed_s
                            _chunks = audio_chunk_count
                            audio_start = None
                            audio_pushed_s = 0.0
                            audio_chunk_count = 0
                            print(f"[turn_complete] {_chunks} chunks ({_pushed_s:.1f}s)")

                            if not listening:
                                # Wait for the robot speaker buffer to finish playing
                                if _started is not None:
                                    remaining = max(0.0, _pushed_s - (time.monotonic() - _started))
                                    if remaining > 0.05:
                                        print(f"[mic] Waiting {remaining:.2f}s for speaker buffer to drain...")
                                        await asyncio.sleep(remaining + 0.5)
                                # Flush any echo that accumulated in the mic queue during TTS
                                flushed = 0
                                try:
                                    while True:
                                        mic_queue.get_nowait()
                                        flushed += 1
                                except Exception:
                                    pass  # queue empty
                                if flushed:
                                    print(f"[mic] Flushed {flushed} stale chunks from queue")
                                listening = True
                                mic_enabled.set()
                                print("[mic open] Mic enabled. Say something or say 'continue' to advance.")
                                restart_idle_timer()
                            else:
                                # After a Q&A answer, restart timer for next input
                                print("[mic open] Answered. Listening for next question or 'continue'.")
                                restart_idle_timer()

                    # --- Function call: user said "continue" / "okay" → advance ---
                    if response.tool_call:
                        print(f"[tool_call] {[c.name for c in response.tool_call.function_calls]}")
                        for call in response.tool_call.function_calls:
                            if call.name == "advance_slide":
                                if idle_task and not idle_task.done():
                                    idle_task.cancel()
                                # Let any buffered audio finish playing before closing
                                if audio_start is not None:
                                    elapsed = time.monotonic() - audio_start
                                    remaining = audio_pushed_s - elapsed
                                    if remaining > 0.1:
                                        await asyncio.sleep(remaining + 0.3)
                                # Acknowledge the function call so Gemini doesn't hang
                                await session.send_tool_response(
                                    function_responses=[types.FunctionResponse(
                                        id=call.id,
                                        name="advance_slide",
                                        response={"status": "advancing"},
                                    )]
                                )
                                done.set()
                                return

            # Run send_mic and receive_audio as cancellable tasks.
            # When done is set (idle timeout or advance_slide), cancel both cleanly
            # so the session closes properly before the next slide begins.
            tasks = [
                asyncio.create_task(send_mic()),
                asyncio.create_task(receive_audio()),
            ]
            await done.wait()
            for t in tasks:
                t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*tasks)

    finally:
        stop_tracking.set()
        with contextlib.suppress(Exception):
            face_thread.join(timeout=2.0)
