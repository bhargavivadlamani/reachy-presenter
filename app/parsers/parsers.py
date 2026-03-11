def parse(file_path: str, parser: str = "pdfplumber") -> list[str]:
    if parser == "pdfplumber":
        return _parse_pdfplumber(file_path)
    elif parser == "python-pptx":
        return _parse_python_pptx(file_path)
    raise ValueError(f"Unknown parser: {parser}")


def _parse_pdfplumber(path: str) -> list[str]:
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


def _parse_python_pptx(path: str) -> list[str]:
    from pptx import Presentation
    prs = Presentation(path)
    slides = []
    for slide in prs.slides:
        text = "\n".join(
            shape.text_frame.text for shape in slide.shapes if shape.has_text_frame
        )
        slides.append(text)
    return slides
