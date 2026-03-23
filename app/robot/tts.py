"""Functions to speak text through Reachy Mini."""

import os
import re
import threading
import time

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from scipy.signal import resample

from app.robot.vision import track_faces_during_speech

load_dotenv()

_openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

OPENAI_TTS_SAMPLERATE = 24000  # OpenAI PCM output is always 24kHz


def _split_sentences(text: str) -> list[str]:
    """Split a script into individual sentences for interruptible playback."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def _speak_chunk(text: str, mini) -> None:
    """Speak a single chunk of text with simultaneous face tracking."""
    # Generate audio as raw 16-bit PCM bytes from OpenAI
    response = _openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
        response_format="pcm",
    )
    pcm_bytes = response.content

    # Convert 16-bit PCM → float32 in range [-1, 1]
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Resample from 24kHz to the robot's output sample rate
    output_sr = mini.media.get_output_audio_samplerate()
    num_output_samples = int(len(samples) * output_sr / OPENAI_TTS_SAMPLERATE)
    samples_resampled = resample(samples, num_output_samples).astype(np.float32)

    duration = len(samples_resampled) / output_sr

    # Thread 1: push audio to speaker (non-blocking call, returns instantly)
    audio_thread = threading.Thread(
        target=mini.media.push_audio_sample, args=(samples_resampled,)
    )
    # Thread 2: track faces for the exact duration of the audio
    movement_thread = threading.Thread(
        target=track_faces_during_speech, args=(mini, duration)
    )

    audio_thread.start()
    movement_thread.start()

    movement_thread.join()  # movement runs for exactly `duration` seconds
    audio_thread.join()


def speak(text: str, mini, qa_handler=None) -> None:
    """Speak text while tracking audience faces for natural eye contact.

    Splits the script into sentences and checks for Q&A interruption between
    each sentence. If qa_handler signals an interrupt, returns immediately so
    the caller can handle the question before resuming.
    """
    sentences = _split_sentences(text)
    for sentence in sentences:
        if qa_handler and qa_handler.is_interrupted():
            return  # Caller (main.py) will call qa_handler.handle_question()
        _speak_chunk(sentence, mini)
