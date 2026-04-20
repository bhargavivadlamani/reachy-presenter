"""Reachy Mini Audience App — entry point for the second robot.

Full pipeline:
  Camera → PresenceDetector → person at booth → AudienceInitiator → greet presenter
  Camera + Stereo mic → AttentionClassifier → SILENT / TO_HUMAN / TO_COMPUTER
  HTTP server → on_slide_presented → react + wait + ask question
  AttentiveListener → natural gaze (toward speaker) + antenna wiggle when listening

Both robots must be running. Set in .env:
  PRESENTER_ROBOT_URL=http://<robot1-ip>:8000
  AUDIENCE_ROBOT_URL=http://<this-robot-ip>:5001  (set on robot 1)
  ROBOT1_YAW_DEG=-35   (degrees; negative = Robot 1 is to the left)
"""

import logging
import threading

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from reachy_mini import ReachyMini, ReachyMiniApp

from app.audience.agent import AudienceAgent
from app.audience.server import AudienceServer
from app.audience.initiator import AudienceInitiator
from app.audience.presence_detector import PresenceDetector
from app.robot.attention_classifier import AttentionClassifier, AttentionClass
from app.robot.attentive_listener import AttentiveListener
from app.robot.gestures import emotion_gesture


def _on_attention_change(cls: AttentionClass) -> None:
    import logging
    logging.getLogger(__name__).info("[main] attention → %s", cls.name)


class ReachyAudienceApp(ReachyMiniApp):
    """AI audience agent with presence detection and attention classification."""

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        # Startup signal — wiggle antennas so the booth knows Robot 2 is awake
        try:
            emotion_gesture(reachy_mini, "excited")
        except Exception:
            pass

        # --- Audience agent (reactions + speech) ---
        agent = AudienceAgent(audience_mini=reachy_mini)
        agent.start()

        # --- HTTP server (receives slide events from presenter) ---
        server = AudienceServer()
        server.register(agent.on_slide_presented)
        server.start()

        # --- Attention classifier (DOA + gaze + VAD) ---
        attention = AttentionClassifier(
            mini=reachy_mini,
            on_change=_on_attention_change,
        )
        attention.start()

        # --- Presence detector (created early so we can access its scanner) ---
        # Callbacks wired after initiator is created below.
        presence = PresenceDetector(mini=reachy_mini)

        # --- Attentive listener (gaze + antennas follow whoever is speaking) ---
        listener = AttentiveListener(
            mini=reachy_mini,
            doa=attention._doa,
            scanner=presence._scanner,
        )
        listener.start()

        # Wrap agent._speak so listener knows when Robot 2's TTS is active.
        # Both initiator and agent.on_slide_presented use this wrapped version.
        _orig_speak = agent._speak

        def _speak(text: str) -> None:
            listener.set_speaking(True)
            try:
                _orig_speak(text)
            finally:
                listener.set_speaking(False)

        agent._speak = _speak

        # --- Initiator (speaks greeting when person arrives) ---
        initiator = AudienceInitiator(
            speak_fn=_speak,
            rms_fn=lambda: attention._doa.last_rms,
            agent=agent,
        )
        initiator.start()

        # Wire presence callbacks now that initiator exists
        presence._on_arrived = initiator.on_person_arrived
        presence._on_left = initiator.on_person_left
        presence.start()

        print("Audience robot ready.")
        print("  Watching for visitors at the booth...")
        print("  Attention classifier running (DOA + gaze + VAD)")
        print("  Slide event server listening on port 5001\n")

        try:
            stop_event.wait()
        finally:
            listener.stop()
            presence.stop()
            initiator.stop()
            attention.stop()
            server.stop()
            agent.stop()


def main() -> None:
    app = ReachyAudienceApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
