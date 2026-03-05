"""Wrapper for OpenAI API calls."""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def generate_script(slide_text: str, model: str = "gpt-4o") -> str:
    """Generate a spoken presentation script for a single slide's text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a robot presenter named Reachy Mini giving a live presentation. "
                    "Given the text content of a slide, generate a natural, engaging spoken script. "
                    "Never use placeholders like [Your Name] or [morning/afternoon/evening] — "
                    "always write complete, ready-to-speak sentences. "
                    "Keep it concise and clear."
                ),
            },
            {
                "role": "user",
                "content": f"Slide content:\n\n{slide_text}",
            },
        ],
    )
    return response.choices[0].message.content.strip()


def classify_slide(slide_text: str) -> str:
    """Classify the emotional tone of a slide for gesture selection.

    Returns one of: excited / neutral / questioning / serious.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the emotional tone of this presentation slide. "
                    "Reply with exactly one word from: excited, neutral, questioning, serious."
                ),
            },
            {"role": "user", "content": slide_text},
        ],
    )
    tone = response.choices[0].message.content.strip().lower()
    return tone if tone in ("excited", "neutral", "questioning", "serious") else "neutral"
