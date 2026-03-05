"""Extract text from PDF pages."""

import pdfplumber


def extract_slides(pdf_path: str) -> list[str]:
    """Return a list of strings, one per page, from the given PDF file."""
    slides = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            slides.append(text.strip())
    return slides
