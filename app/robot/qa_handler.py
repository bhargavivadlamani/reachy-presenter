"""Keyboard-triggered Q&A interruption during live presentations.

Press SPACEBAR at any time during speech to pause Reachy and ask a question.
The question is answered using the full presentation document as context.
"""

import threading

try:
    from pynput import keyboard as pynput_keyboard
    _PYNPUT_AVAILABLE = True
except ImportError:
    _PYNPUT_AVAILABLE = False


class QAHandler:
    """Listens for spacebar press to interrupt speech and handle audience questions.

    Usage:
        qa = QAHandler()
        qa.start_listening()
        # ... speak slides, passing qa to speak() ...
        # If interrupted, call qa.handle_question(mini, document_text)
        qa.stop_listening()
    """

    def __init__(self):
        self._interrupt = threading.Event()
        self._listener = None

    def start_listening(self) -> None:
        """Start background keyboard listener — spacebar triggers interrupt."""
        self._interrupt.clear()
        if not _PYNPUT_AVAILABLE:
            print("[QA] pynput not available — Q&A interruption disabled.")
            return
        self._listener = pynput_keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key) -> bool | None:
        if key == pynput_keyboard.Key.space:
            self._interrupt.set()
            return False  # Stop this listener; re-armed after Q&A

    def stop_listening(self) -> None:
        if self._listener and self._listener.is_alive():
            self._listener.stop()

    def is_interrupted(self) -> bool:
        return self._interrupt.is_set()

    def handle_question(self, mini, document_text: str) -> None:
        """Collect audience question, speak answer using document context, re-arm."""
        from app.llm.gemini_client import answer_question
        from app.robot.tts import speak

        print("\n[Q&A] Presentation paused.")
        question = input("Type the audience question: ").strip()

        if question:
            print("Searching document for answer...")
            answer = answer_question(question, document_text)
            print(f"Reachy: {answer}\n")
            speak(answer, mini)

        self._interrupt.clear()
        self.start_listening()  # Re-arm for the next question
