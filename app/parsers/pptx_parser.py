"""Extract text from PPTX slides."""

from pptx import Presentation


def extract_slides(pptx_path: str) -> list[str]:
    """Return a list of strings, one per slide, from the given PPTX file."""
    prs = Presentation(pptx_path)
    slides = []
    for slide in prs.slides:
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = " ".join(run.text for run in para.runs).strip()
                    if line:
                        texts.append(line)
        slides.append("\n".join(texts))
    return slides
