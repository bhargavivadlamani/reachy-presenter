"""Script generation via FAU TRUSSED proxy (OpenAI-compatible, gpt-4o vision).

Switched from direct Google API (rate-limited) to TRUSSED OpenAI endpoint.
Same interface — generate_script(), classify_slide() — unchanged for main.py.
"""

import base64
import io
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://fauengtrussed.fau.edu/provider/generic",
)
_MODEL = "gpt-4o"


def _img_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def generate_script(slide_image) -> str:
    """Generate a spoken presentation script by visually reading a slide image."""
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{_img_b64(slide_image)}"}},
                {"type": "text", "text": (
                    "You are writing a script for a live presentation. "
                    "Look at this slide and write a natural, conversational spoken script "
                    "as if a confident human presenter is delivering it live to an audience. "
                    "Use first-person language and speak directly to the audience. "
                    "Describe any charts, diagrams, or visuals as you would out loud. "
                    "Vary your sentence rhythm — mix short punchy sentences with longer ones. "
                    "Never use placeholder text or stage directions. "
                    "Write exactly what should be spoken, nothing else. Keep it to 4-6 sentences."
                )},
            ],
        }],
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def classify_slide(slide_image) -> str:
    """Classify the emotional tone of a slide. Returns: excited/neutral/questioning/serious."""
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{_img_b64(slide_image)}"}},
                {"type": "text", "text": (
                    "Look at this presentation slide and classify its emotional tone. "
                    "Reply with exactly one word from: excited, neutral, questioning, serious."
                )},
            ],
        }],
        max_tokens=5,
    )
    tone = response.choices[0].message.content.strip().lower()
    return tone if tone in ("excited", "neutral", "questioning", "serious") else "neutral"


def transcribe_audio(wav_bytes: bytes) -> str:
    """Transcribe spoken audio — fallback, Live API handles this natively."""
    response = _client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.wav", wav_bytes, "audio/wav"),
    )
    return response.text.strip()


def answer_question(question: str, document_text: str) -> str:
    """Answer an audience question — fallback, Live API handles this natively."""
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "You are a robot presenter named Reachy Mini answering an audience question live. "
                "Based on the presentation content below, answer in 2-3 sentences for spoken delivery.\n\n"
                f"Presentation content:\n{document_text}\n\nQuestion: {question}"
            ),
        }],
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()
