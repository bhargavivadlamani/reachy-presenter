"""Minimal Gemini Live bidi test — audio in/out, no robot SDK.

Run:  python test_live.py
Say anything. Should interrupt mid-response. Say 'stop' or Ctrl+C to exit.
"""

import asyncio
import os
import threading

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from scipy.signal import resample as scipy_resample

load_dotenv()

MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
MIC_SR = 16000   # Gemini expects 16kHz input
GEM_SR = 24000   # Gemini outputs 24kHz

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
    http_options={"api_version": "v1beta"},
)


async def run():
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction="You are a helpful assistant. Respond conversationally and concisely.",
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                prefix_padding_ms=200,
                silence_duration_ms=600,
            )
        ),
    )

    print(f"Connecting to {MODEL}...")
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        print("Connected! Say something. Ctrl+C to stop.\n")

        # Output: write directly to sounddevice output stream
        out_sr = sd.query_devices(kind="output")["default_samplerate"]
        out_sr = int(out_sr)
        out_stream = sd.OutputStream(samplerate=out_sr, channels=1, dtype="float32")
        out_stream.start()

        mic_queue: asyncio.Queue[bytes] = asyncio.Queue()

        def mic_reader():
            chunk = int(MIC_SR * 0.1)  # 100ms
            with sd.InputStream(samplerate=MIC_SR, channels=1, dtype="int16", blocksize=chunk) as stream:
                while not stop.is_set():
                    data, _ = stream.read(chunk)
                    try:
                        loop.call_soon_threadsafe(mic_queue.put_nowait, data.tobytes())
                    except RuntimeError:
                        break

        threading.Thread(target=mic_reader, daemon=True).start()

        async def send_mic():
            while not stop.is_set():
                try:
                    data = await asyncio.wait_for(mic_queue.get(), timeout=0.5)
                    await session.send_realtime_input(
                        audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                    )
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    print(f"[send_mic] {e}")
                    break

        async def receive_and_play():
            chunk_count = 0
            async for response in session.receive():
                if response.data:
                    samples = (
                        np.frombuffer(response.data, dtype=np.int16)
                        .astype(np.float32) / 32768.0
                    )
                    if out_sr != GEM_SR:
                        n = int(len(samples) * out_sr / GEM_SR)
                        samples = scipy_resample(samples, n).astype(np.float32)
                    if chunk_count == 0:
                        print("[audio] Gemini speaking...")
                    chunk_count += 1
                    out_stream.write(samples)

                sc = response.server_content
                if sc:
                    if sc.interrupted:
                        print(f"[interrupted] after {chunk_count} chunks")
                        chunk_count = 0
                    if sc.turn_complete:
                        print(f"[turn_complete] {chunk_count} chunks. Listening...")
                        chunk_count = 0

        tasks = [
            asyncio.create_task(send_mic()),
            asyncio.create_task(receive_and_play()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            stop.set()
            for t in tasks:
                t.cancel()
            out_stream.stop()
            out_stream.close()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nStopped.")
