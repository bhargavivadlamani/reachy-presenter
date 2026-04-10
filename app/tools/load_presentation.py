"""Tool: load_presentation.

Parses a PDF or PPTX file into slide images, then generates a spoken script for
each slide in a background thread so the agent stays responsive while loading.
Scripts become available one by one; present_slide() checks readiness per slide.
"""

import os
import threading
import time
from typing import Optional

_scripts: list[str] = []
_document_text: str = ""
_total_slides: int = 0
_generating: bool = False
_collection_name: str = ""
_lock = threading.Lock()


def get_slide_script(slide_number: int) -> Optional[str]:
    with _lock:
        if 1 <= slide_number <= len(_scripts):
            return _scripts[slide_number - 1]
    return None


def get_document_text() -> str:
    with _lock:
        return _document_text


def get_collection_name() -> str:
    with _lock:
        return _collection_name


def get_slide_count() -> int:
    with _lock:
        return len(_scripts)


def get_total_slides() -> int:
    with _lock:
        return _total_slides


def is_generating() -> bool:
    return _generating


def _generate_background(images: list, basename: str) -> None:
    global _document_text, _generating
    from app.tools.generate_script import generate_script

    for i, image in enumerate(images, start=1):
        print(f"[load_presentation] Slide {i}/{len(images)} — generating script...")
        script = generate_script(image)
        with _lock:
            _scripts.append(script)
        print(f"[load_presentation] Slide {i} ready.")
        if i < len(images):
            time.sleep(7)

    with _lock:
        _document_text = "\n\n".join(
            f"Slide {i}: {s}" for i, s in enumerate(_scripts, start=1)
        )
    _generating = False
    print(f"[load_presentation] All {len(images)} slides ready.")


def load_presentation(file_path: str) -> str:
    """Load a PDF or PPTX presentation and generate spoken scripts in the background.

    Returns immediately so the agent can keep chatting while scripts are prepared.
    Scripts become available slide by slide; present_slide() will report if a
    requested slide isn't ready yet.

    Args:
        file_path: Absolute or relative path to a .pdf or .pptx file.

    Returns:
        Confirmation that background generation has started.
    """
    global _scripts, _document_text, _total_slides, _generating, _collection_name

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        from app.parsers.pdf_parser import extract_slide_images
    elif ext in (".pptx", ".ppt"):
        from app.parsers.pptx_parser import extract_slide_images
    else:
        return f"Unsupported file type '{ext}'. Provide a .pdf or .pptx file."

    try:
        images = extract_slide_images(file_path)
    except Exception as e:
        return f"Failed to parse file: {e}"

    with _lock:
        _scripts = []
        _document_text = ""
        _total_slides = len(images)
        _collection_name = os.path.splitext(os.path.basename(file_path))[0].lower()
    _generating = True

    threading.Thread(
        target=_generate_background,
        args=(images, os.path.basename(file_path)),
        daemon=True,
    ).start()

    return (
        f"Started preparing {len(images)} slides from {os.path.basename(file_path)}. "
        f"Slide 1 will be ready in roughly 10 seconds, each following slide about 7 seconds after that. "
        f"Feel free to keep talking — just say 'start' or 'begin' when you want to kick off the presentation."
    )
