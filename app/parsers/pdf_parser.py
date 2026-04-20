"""Extract text and images from PDF pages."""

import pdfplumber
from pdf2image import convert_from_path


def extract_slides(pdf_path: str) -> list[str]:
    """Return a list of strings, one per page, from the given PDF file."""
    slides = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            slides.append(text.strip())
    return slides


def extract_slide_images(pdf_path: str) -> list:
    """Return a list of PIL Images, one per page, rendered at 150 DPI."""
    return convert_from_path(pdf_path, dpi=150)
