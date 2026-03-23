"""Extract text and images from PPTX slides."""

import subprocess
import tempfile
from pathlib import Path

from pdf2image import convert_from_path
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


def extract_slide_images(pptx_path: str) -> list:
    """Return a list of PIL Images, one per slide.

    Converts PPTX → PDF via LibreOffice (must be installed), then renders pages.
    On robot: sudo apt-get install libreoffice poppler-utils
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", tmpdir, pptx_path],
            check=True,
            capture_output=True,
        )
        pdf_path = Path(tmpdir) / (Path(pptx_path).stem + ".pdf")
        return convert_from_path(str(pdf_path), dpi=150)
