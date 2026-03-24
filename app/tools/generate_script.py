"""Helper: generate_script.

Not an ADK agent tool — takes a PIL image which the agent cannot supply directly.
Used by main.py in the pre-generation loop before the presentation session starts.

Source of truth: app/llm/gemini_client.py
"""

from app.llm.gemini_client import generate_script

__all__ = ["generate_script"]
