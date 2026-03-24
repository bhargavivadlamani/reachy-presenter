"""Audio format constants and PCM conversion helpers.

Gemini Live API contract:
  - Input:  16kHz int16 PCM  (mic -> Gemini)
  - Output: 24kHz int16 PCM  (Gemini -> speaker)
"""

import numpy as np
from scipy.signal import resample as scipy_resample

MIC_SR = 16000   # Gemini Live expects 16kHz int16 PCM input
GEM_SR = 24000   # Gemini Live outputs 24kHz int16 PCM
VAD_THRESHOLD = 0.02  # RMS amplitude to treat as speech (tune if too sensitive)


def to_pcm_bytes(frame: np.ndarray, src_sr: int) -> bytes:
    """Convert a mic frame (float32, any SR) to 16kHz int16 PCM bytes."""
    if frame.ndim == 2:
        frame = frame.mean(axis=1)
    frame = frame.astype(np.float32)
    if src_sr != MIC_SR:
        n = int(len(frame) * MIC_SR / src_sr)
        frame = scipy_resample(frame, n).astype(np.float32)
    return (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def decode_pcm(raw_bytes: bytes) -> np.ndarray:
    """Decode Gemini's 24kHz int16 PCM bytes to float32 samples."""
    return np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
