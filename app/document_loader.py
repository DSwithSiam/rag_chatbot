from pathlib import Path

from docx import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def validate_extension(file_name: str) -> None:
    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only PDF and DOCX files are supported.")


def read_document(file_path: Path) -> str:
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        return _read_pdf(file_path)
    if extension == ".docx":
        return _read_docx(file_path)
    raise ValueError("Only PDF and DOCX files are supported.")


def _read_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append((page.extract_text() or "").strip())
    text = "\n".join(part for part in parts if part)
    if not text:
        raise ValueError("No readable text found in PDF.")
    return text


def _read_docx(file_path: Path) -> str:
    doc = Document(str(file_path))
    text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text and p.text.strip())
    if not text:
        raise ValueError("No readable text found in DOCX.")
    return text
