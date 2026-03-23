"""Orchestrates parsing, LLM calls and presentation through Reachy Mini."""

import argparse
import os
import time

from reachy_mini import ReachyMini

from app.llm.gemini_client import generate_script
from app.robot.gestures import slide_transition
from app.robot.live_presenter import present_slide


def load_slide_images(file_path: str) -> list:
    """Return slide images (PIL) for vision-based Gemini processing."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        from app.parsers.pdf_parser import extract_slide_images
    elif ext in (".pptx", ".ppt"):
        from app.parsers.pptx_parser import extract_slide_images
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return extract_slide_images(file_path)


def present(file_path: str) -> None:
    print("Rendering slides as images...")
    slide_images = load_slide_images(file_path)
    print(f"  {len(slide_images)} slides found.\n")

    # Pre-generate all scripts using Gemini Vision (one call per slide)
    print("Generating scripts for all slides (Gemini Vision)...")
    scripts = []
    for i, image in enumerate(slide_images, start=1):
        print(f"  Slide {i}: generating...")
        scripts.append(generate_script(image))
        if i < len(slide_images):
            time.sleep(7)  # TRUSSED rate limit: 10 req/min → wait 7s between calls

    # Q&A context: joined scripts contain Gemini's visual descriptions
    document_text = "\n\n".join(
        f"Slide {i}: {s}" for i, s in enumerate(scripts, start=1) if s
    )
    print("All scripts ready. Starting presentation...\n")
    print("Ask questions any time. Say 'continue' or 'okay' to advance slides.\n")

    with ReachyMini() as mini:
        mini.media.start_playing()
        mini.media.start_recording()   # required before get_audio_sample() returns data
        try:
            for i, script in enumerate(scripts, start=1):
                print(f"--- Slide {i} ---")
                print(f"{script}\n")
                slide_transition(mini)                       # glance down → look up (~1.2s)
                present_slide(mini, script, document_text)  # Gemini Live: speak + Q&A

        except KeyboardInterrupt:
            print("\nPresentation stopped.")
        finally:
            mini.media.stop_recording()
            mini.media.stop_playing()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reachy Mini Presenter")
    parser.add_argument("file", help="Path to a .pdf or .pptx file")
    args = parser.parse_args()
    present(args.file)


if __name__ == "__main__":
    main()
