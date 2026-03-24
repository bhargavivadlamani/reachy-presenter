"""Tool: load_presentation.

Parses a PDF or PPTX file into slide images, generates a spoken script for
each slide via vision LLM, and stores the result in module-level state so
present_slide() can reference slides by number.
"""

import os
import time
from typing import Optional

_scripts: list[str] = []
_document_text: str = ""


def get_slide_script(slide_number: int) -> Optional[str]:
    if 1 <= slide_number <= len(_scripts):
        return _scripts[slide_number - 1]
    return None


def get_document_text() -> str:
    return _document_text


def get_slide_count() -> int:
    return len(_scripts)


def load_presentation(file_path: str) -> str:
    """Load a PDF or PPTX presentation and generate a spoken script for each slide.

    Call this when the user asks to load, open, or present a file.
    After loading, present individual slides with present_slide(slide_number=N).

    Args:
        file_path: Absolute or relative path to a .pdf or .pptx file.

    Returns:
        Summary listing all slides and a preview of each script.
    """
    global _scripts, _document_text

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        from app.parsers.pdf_parser import extract_slide_images
    elif ext in (".pptx", ".ppt"):
        from app.parsers.pptx_parser import extract_slide_images
    else:
        return f"Unsupported file type '{ext}'. Provide a .pdf or .pptx file."

    from app.llm.gemini_client import generate_script

    try:
        images = extract_slide_images(file_path)
    except Exception as e:
        return f"Failed to parse file: {e}"

    scripts = []
    for i, image in enumerate(images, start=1):
        print(f"[load_presentation] Slide {i}/{len(images)} — generating script...")
        scripts.append(generate_script(image))
        if i < len(images):
            time.sleep(7)  # FAU TRUSSED rate limit: ~10 req/min

    _scripts = scripts
    _document_text = "\n\n".join(
        f"Slide {i}: {s}" for i, s in enumerate(scripts, start=1)
    )

    lines = [f"Loaded {len(scripts)} slides from {os.path.basename(file_path)}.\n"]
    for i, s in enumerate(scripts, start=1):
        preview = s[:100].replace("\n", " ")
        lines.append(f"  Slide {i}: {preview}...")
    return "\n".join(lines)
