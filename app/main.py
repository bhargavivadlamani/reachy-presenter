"""Reachy Mini Presenter App — entry point for the Reachy Mini app ecosystem."""

import threading

from reachy_mini import ReachyMini, ReachyMiniApp

from app.agent import run_for_robot
from app.tools.present_slide import set_mini
from app.robot.idle_behavior import IdleBehavior
from app.robot.gestures import set_idle_behavior


class ReachyPresenterApp(ReachyMiniApp):
    """AI-powered presenter agent using Gemini Bidi streaming.

    Loads PDF/PPTX presentations, generates spoken scripts, performs
    expressive gestures, and answers audience questions via RAG.
    """

    request_media_backend: str | None = None  # use default (GStreamer on robot)

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the presenter agent until stop_event is set."""
        set_mini(reachy_mini)
        idle = IdleBehavior(reachy_mini)
        set_idle_behavior(idle)
        idle.start()
        print("Reachy Presenter ready. Just start talking.\n")
        try:
            run_for_robot(reachy_mini)
        finally:
            idle.stop()


def main() -> None:
    """CLI entry point: reachy-presenter command."""
    app = ReachyPresenterApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
