"""ADK tool: present_slide.

Registered with the agent. Performs a slide transition gesture on the robot
and signals the agent to read the script aloud via its own bidi TTS stream.

Call set_mini(mini) before starting the audio session to wire up the robot.
"""

_mini_ref = None


def set_mini(mini) -> None:
    global _mini_ref
    _mini_ref = mini


def present_slide(script: str = "", slide_number: int = 0, document_text: str = "") -> str:
    """Perform a slide transition gesture and prepare to deliver a slide.

    Call this when the user asks to present a slide. Use slide_number when a
    presentation has been loaded with load_presentation(); use script for
    ad-hoc content. After this returns, read the script aloud word for word.

    Args:
        script: Spoken script to deliver. Ignored if slide_number is given.
        slide_number: 1-based slide index from a loaded presentation.
        document_text: Slide context for Q&A. Auto-filled from loaded presentation.

    Returns:
        Confirmation with the script to read aloud.
    """
    if slide_number > 0:
        from app.tools.load_presentation import (
            get_slide_script, get_document_text, get_slide_count,
            get_total_slides, is_generating,
        )
        fetched = get_slide_script(slide_number)
        if fetched is None:
            ready, total = get_slide_count(), get_total_slides()
            if is_generating():
                return (
                    f"Slide {slide_number} isn't ready yet — {ready} of {total} scripts generated so far. "
                    f"Wait a few seconds and try again."
                )
            elif total == 0:
                return "No presentation has been loaded yet."
            else:
                return f"Slide {slide_number} not found — only {ready} slides are loaded."
        script = fetched
        if not document_text:
            document_text = get_document_text()

    if _mini_ref is not None:
        try:
            from app.robot.gestures import slide_transition
            slide_transition(_mini_ref)
        except Exception as e:
            print(f"[gesture] {e}")

    if not script:
        return "No script provided and no presentation loaded. Ask the user what to say."

    from app.audience.notifier import notify_slide_presented
    notify_slide_presented(slide_number, script)

    return f"Slide transition complete. Now read this script aloud: {script}"
