"""Reachy Mini Audience App — entry point for the second robot.

Pipeline:
  Boot → BidiConversationSession starts (persistent, auto-restarts)
  Camera → PresenceDetector → visitor arrives → greet_visitor() + set_visitor_present(True)
                            → visitor leaves  → set_visitor_present(False) only (conv continues)
  AttentiveListener → visitor present: look at Robot 1 during silence, DOA during speech
                    → no visitor: resume scan
  HTTP /slide → mute_output() → AudienceAgent speaks → unmute_output()

.env required:
  PRESENTER_ROBOT_URL=http://<robot1-ip>:8000
  AUDIENCE_ROBOT_URL=http://<this-robot-ip>:5001  (set on robot 1)
  ROBOT1_YAW_DEG=-35  (negative = Robot 1 is to the left of Robot 2)
"""

import logging
import threading

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from reachy_mini import ReachyMini, ReachyMiniApp

from app.audience.agent import AudienceAgent
from app.audience.server import AudienceServer
from app.audience.bidi_conversation import BidiConversationSession
from app.audience.presence_detector import PresenceDetector
from app.robot.attention_classifier import AttentionClassifier, AttentionClass
from app.robot.attentive_listener import AttentiveListener
from app.robot.gestures import emotion_gesture


def _on_attention_change(cls: AttentionClass) -> None:
    logging.getLogger(__name__).info("[main] attention → %s", cls.name)


class ReachyAudienceApp(ReachyMiniApp):

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        try:
            emotion_gesture(reachy_mini, "excited")
        except Exception:
            pass

        # --- Slide-reaction agent (TTS questions between slides) ---
        agent = AudienceAgent(audience_mini=reachy_mini)
        agent.start()

        # --- Attention classifier (DOA + gaze + VAD) ---
        attention = AttentionClassifier(
            mini=reachy_mini,
            on_change=_on_attention_change,
        )
        attention.start()

        # --- Presence detector (creates ScanBehavior internally) ---
        presence = PresenceDetector(mini=reachy_mini)

        # --- Attentive listener: looks at Robot 1 when visitor present, scans otherwise ---
        listener = AttentiveListener(
            mini=reachy_mini,
            doa=attention._doa,
            scanner=presence.scanner,
        )
        listener.start()

        # --- Persistent Bidi conversation (starts at boot, never stops on visitor leave) ---
        conv = BidiConversationSession(on_speaking_changed=listener.set_speaking)

        # Wire presence callbacks
        presence._on_arrived = lambda: _on_arrived(reachy_mini, conv, listener, agent)
        presence._on_left = lambda: _on_left(conv, listener)

        # --- HTTP server: mute Bidi output while AudienceAgent handles slide audio ---
        def _on_slide(slide_number: int, script: str) -> None:
            conv.mute_output()
            agent.on_slide_presented(slide_number, script)
            # Wait for the slide reaction thread to finish, then restore Bidi audio
            def _wait_and_unmute() -> None:
                import time
                while agent.busy:
                    time.sleep(0.2)
                conv.unmute_output()
            threading.Thread(target=_wait_and_unmute, daemon=True).start()

        server = AudienceServer()
        server.register(_on_slide)
        server.start()

        # Start persistent conversation session
        conv.start(reachy_mini)

        presence.start()

        print("Audience robot ready.")
        print("  Bidi conversation: always-on Gemini Live")
        print("  AttentiveListener: looks at Robot 1 when visitor present")
        print("  Slide server: port 5001 (mutes Bidi during slide TTS)\n")

        try:
            stop_event.wait()
        finally:
            conv.shutdown()
            presence.stop()
            listener.stop()
            attention.stop()
            server.stop()
            agent.stop()


def _on_arrived(mini, conv: BidiConversationSession,
                listener: AttentiveListener, agent: AudienceAgent) -> None:
    listener.set_visitor_present(True)
    if not agent.busy:
        conv.greet_visitor()


def _on_left(conv: BidiConversationSession, listener: AttentiveListener) -> None:
    # Visitor gone — stop tracking them, but conversation keeps running
    listener.set_visitor_present(False)


def main() -> None:
    app = ReachyAudienceApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
