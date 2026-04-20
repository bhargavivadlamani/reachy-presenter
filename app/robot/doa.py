"""GCC-PHAT Direction of Arrival estimation from stereo microphone.

Reads stereo audio from the Reachy Mini USB mic, applies the Generalized
Cross-Correlation Phase Transform algorithm to estimate which horizontal
direction sound is coming from.

Output: angle in degrees
  0°   = directly in front of robot
  -90° = hard left
  +90° = hard right

Tune MIC_DISTANCE_M if results seem off — measure actual mic separation
on the USB audio device hardware.
"""

import logging
import threading
from typing import Optional, Callable

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE   = 16000
CHUNK_SIZE    = 1600          # 100ms windows
MIC_DIST_M    = 0.06          # metres between mics — tune for actual hardware
SPEED_OF_SOUND = 343.0
MAX_TDOA      = MIC_DIST_M / SPEED_OF_SOUND
VAD_RMS_MIN   = 0.005         # ignore frames quieter than this


def gcc_phat(sig_l: np.ndarray, sig_r: np.ndarray) -> float:
    """Return estimated angle (degrees) of dominant sound source."""
    n = len(sig_l) + len(sig_r) - 1
    fft_size = 1 << (n - 1).bit_length()

    SL = np.fft.rfft(sig_l, n=fft_size)
    SR = np.fft.rfft(sig_r, n=fft_size)

    cross = SL * np.conj(SR)
    denom = np.abs(cross)
    denom[denom < 1e-10] = 1e-10
    cc = np.fft.irfft(cross / denom, n=fft_size)

    max_lag = int(np.ceil(MAX_TDOA * SAMPLE_RATE)) + 1
    # GCC-PHAT peak search in valid TDOA range
    candidates = np.concatenate([cc[:max_lag], cc[-max_lag:]])
    peak = np.argmax(candidates)
    if peak < max_lag:
        lag = peak
    else:
        lag = peak - len(candidates)

    tdoa = np.clip(lag / SAMPLE_RATE, -MAX_TDOA, MAX_TDOA)
    sin_val = np.clip((tdoa * SPEED_OF_SOUND) / MIC_DIST_M, -1.0, 1.0)
    return float(np.degrees(np.arcsin(sin_val)))


class DOAEstimator:
    """Continuously reads stereo mic and estimates sound direction."""

    def __init__(self, on_angle: Optional[Callable[[float], None]] = None) -> None:
        self._on_angle = on_angle
        self._angle = 0.0
        self._last_rms = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("[DOA] started — mic dist=%.0fcm, max_tdoa=%.2fms",
                    MIC_DIST_M * 100, MAX_TDOA * 1000)

    def stop(self) -> None:
        self._stop.set()

    @property
    def angle(self) -> float:
        with self._lock:
            return self._angle

    @property
    def last_rms(self) -> float:
        with self._lock:
            return self._last_rms

    def _run(self) -> None:
        # Try stereo sounddevice first; fall back gracefully if mic is busy
        # (GStreamer may already own the ALSA device)
        for device in [0, 7, None]:
            try:
                kwargs = dict(
                    samplerate=SAMPLE_RATE,
                    channels=2,
                    dtype="float32",
                    blocksize=CHUNK_SIZE,
                )
                if device is not None:
                    kwargs["device"] = device
                with sd.InputStream(**kwargs) as stream:
                    logger.info("[DOA] stream open on device %s", device)
                    while not self._stop.is_set():
                        data, _ = stream.read(CHUNK_SIZE)
                        if data.shape[1] < 2:
                            continue
                        sig_l = data[:, 0].copy()
                        sig_r = data[:, 1].copy()
                        rms = float(np.sqrt(np.mean(sig_l ** 2 + sig_r ** 2) / 2))
                        with self._lock:
                            self._last_rms = rms
                        if rms < VAD_RMS_MIN:
                            continue
                        angle = gcc_phat(sig_l, sig_r)
                        with self._lock:
                            self._angle = angle
                        logger.debug("[DOA] rms=%.4f angle=%.1f°", rms, angle)
                        if self._on_angle:
                            self._on_angle(angle)
                    return   # clean exit
            except Exception as e:
                logger.warning("[DOA] device %s failed: %s", device, e)
        logger.warning("[DOA] all devices failed — DOA disabled, VAD-only mode")
