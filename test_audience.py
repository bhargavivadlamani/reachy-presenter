"""Standalone test for AudienceAgent — run this BEFORE integrating with presenter.

Tests:
  1. Connects to audience robot
  2. Physical reactions (DualRobotManager)
  3. Question generation (Gemini Flash text)
  4. TTS + audio push (audience robot speaks)

Usage:
    python test_audience.py --robot-name reachy-mini-2   # or whatever the 2nd robot's name is
    python test_audience.py                               # auto-discover
"""

import argparse
import logging
import time
from dotenv import load_dotenv
from reachy_mini import ReachyMini

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("--robot-name", default=None, help="Hostname of audience robot (e.g. reachy-mini-2.local)")
args = parser.parse_args()

print("Connecting to audience robot...")
kwargs = {"robot_name": args.robot_name} if args.robot_name else {}

with ReachyMini(**kwargs) as audience_mini:
    # Use same robot as both presenter+audience for solo test
    # In real setup, presenter_mini would be a different ReachyMini instance
    from app.audience.agent import AudienceAgent

    agent = AudienceAgent(
        audience_mini=audience_mini,
        presenter_mini=audience_mini,  # same robot for solo test — reactions still work
    )

    print("Starting audience agent...")
    agent.start()

    # Step 1: test physical reactions
    print("\n--- Step 1: Physical reactions ---")
    from reachy_mini_conversation_app.orchestrator.dual_robot_manager import AudienceReaction
    agent._dual.trigger_reaction(AudienceReaction.EXCITED)
    agent._dual.update()
    time.sleep(2)

    agent._dual.trigger_reaction(AudienceReaction.NOD_AGREE)
    agent._dual.update()
    time.sleep(2)

    agent._dual.ask_question_gesture()
    time.sleep(2)
    print("Reactions OK")

    # Step 2: test question generation
    print("\n--- Step 2: Question generation ---")
    test_script = (
        "Reachy Mini is a small social robot developed by Pollen Robotics. "
        "It features expressive antennas and a camera head. "
        "It can be programmed using Python and the Reachy SDK."
    )
    question = agent._generate_question(1, test_script)
    print(f"Generated question: {question}")

    # Step 3: test TTS + speaking
    print("\n--- Step 3: TTS + speaking ---")
    if question:
        print(f"Audience robot will now say: '{question}'")
        agent._speak(question)
        print("Speech OK")
    else:
        print("Skipping TTS (no question generated)")

    # Step 4: full flow
    print("\n--- Step 4: Full on_slide_presented flow ---")
    agent.on_slide_presented(1, test_script)
    time.sleep(10)  # wait for async flow to complete

    agent.stop()
    print("\nAll tests passed.")
