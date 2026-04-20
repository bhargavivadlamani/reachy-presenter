"""Reachy Mini Presenter App — entry point for the Reachy Mini app ecosystem.

Unified app combining:
- Gemini Bidi conversational agent (presenter + Q&A + RAG)
- Greetings integration: face tracking (Reachy looks at you) + hand gesture reactions
- Attention gate: sd_attention SDK filters audio so Reachy only responds when
  spoken to directly (gracefully skipped if sd_attention is not installed)
"""

import threading

from reachy_mini import ReachyMini, ReachyMiniApp

from app.agent import run_for_robot
from app.tools.present_slide import set_mini
from app.robot.idle_behavior import IdleBehavior
from app.robot.gestures import set_idle_behavior
from app.robot.greetings_integration import GreetingsIntegration
from app.robot.attention_gate import AttentionGate


class ReachyPresenterApp(ReachyMiniApp):
    """AI-powered presenter + greeter agent using Gemini Bidi streaming.

    Features:
    - Conversational AI (chat, Q&A, presentations, RAG)
    - Attention-gated listening: only responds when spoken to directly
      (requires sd_attention wheel; falls back to always-on if not installed)
    - Face tracking: Reachy follows your face while listening
    - Hand gesture reactions: wave → welcoming, thumbs up → proud, etc.
    - Dance and emotion tools (on request)
    """

    request_media_backend: str | None = None  # use default (GStreamer on robot)

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the unified presenter + greetings agent."""
        set_mini(reachy_mini)

        idle = IdleBehavior(reachy_mini)
        set_idle_behavior(idle)
        idle.start()

        greetings = GreetingsIntegration(reachy_mini, idle)
        greetings.start()

        # Attention gate — filters audio to Gemini based on sd_attention
        gate = AttentionGate()
        if gate.setup(reachy_mini):
            print("Attention detection active — Reachy will only respond when spoken to directly.")
        else:
            print("Always-on mode (install sd_attention wheel to enable attention filtering).")

        print("Reachy ready. Just start talking — I can present, chat, dance, and greet!\n")
        try:
            run_for_robot(reachy_mini, attention_gate=gate)
        finally:
            gate.stop()
            greetings.stop()
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
