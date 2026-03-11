"""Orchestrates parsing, LLM call and presentation through Reachy Mini."""

import argparse

from reachy_mini import ReachyMini

from app.llm.openai_client import classify_slide, generate_script
from app.robot.gestures import emotion_gesture, slide_transition
from app.robot.tts import speak


def present(file_path: str, parser: str = "pdfplumber") -> None:
    from app.parsers.parsers import parse
    slides = parse(file_path, parser=parser)

    # Pre-generate all scripts and classify emotions before connecting to robot
    print("Generating scripts for all slides...")
    scripts, emotions = [], []
    for i, slide_text in enumerate(slides, start=1):
        if not slide_text.strip():
            print(f"  Slide {i}: (empty, will skip)")
            scripts.append(None)
            emotions.append(None)
        else:
            print(f"  Slide {i}: generating...")
            scripts.append(generate_script(slide_text))
            emotions.append(classify_slide(slide_text))
    print("All scripts ready.\n")

    with ReachyMini() as mini:
        mini.media.start_playing()
        try:
            for i, (script, emotion) in enumerate(zip(scripts, emotions), start=1):
                if script is None:
                    continue
                print(f"--- Slide {i} [{emotion}] ---")
                print(f"{script}\n")
                input("Press Enter to speak this slide (or Ctrl+C to stop)...")
                slide_transition(mini)       # glance down → look up (~1.2s)
                emotion_gesture(mini, emotion)  # expressive move (~0.6-1.1s)
                speak(script, mini)          # speak + face track simultaneously
        except KeyboardInterrupt:
            print("\nPresentation stopped.")
        finally:
            mini.media.stop_playing()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reachy Mini Presenter")
    parser.add_argument("file", help="Path to a .pdf or .pptx file")
    parser.add_argument("--parser", default="pdfplumber")
    args = parser.parse_args()
    present(args.file, parser=args.parser)


if __name__ == "__main__":
    main()
