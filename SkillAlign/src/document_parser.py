"""Local resume parsing for PDF, DOCX and TXT. Nothing is sent to any external service."""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .text_preprocessor import normalize_whitespace

MAX_FILE_BYTES = 5 * 1024 * 1024
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")


class DocumentParseError(Exception):
    """Raised with a user-friendly message when a document cannot be read."""


@dataclass
class ParsedDocument:
    text: str
    kind: str
    filename: str
    pages: int | None = None
    warnings: list[str] = field(default_factory=list)


def _parse_txt(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ["utf-16"]
    else:
        encodings = ["utf-8-sig", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise DocumentParseError("The text file could not be decoded. Please save it as UTF-8 and try again.")


def _parse_pdf(data: bytes) -> tuple[str, int]:
    if not data.startswith(b"%PDF"):
        raise DocumentParseError("This file has a .pdf extension but is not a valid PDF document.")
    try:
        import pymupdf as fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DocumentParseError("PDF support needs PyMuPDF. Install it with: pip install pymupdf") from exc
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentParseError(f"The PDF could not be opened ({type(exc).__name__}). It may be corrupted.") from exc
    try:
        if doc.needs_pass:
            raise DocumentParseError("The PDF is password-protected. Remove the password and upload it again.")
        pages = [page.get_text("text") for page in doc]
        return "\n".join(pages), len(pages)
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(f"Text extraction from the PDF failed ({type(exc).__name__}).") from exc
    finally:
        doc.close()


def _parse_docx(data: bytes) -> str:
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise DocumentParseError("This file has a .docx extension but is not a valid Word document.")
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover
        raise DocumentParseError("DOCX support needs python-docx. Install it with: pip install python-docx") from exc
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentParseError(f"The Word document could not be read ({type(exc).__name__}).") from exc
    lines = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            cells = []
            for cell in row.cells:
                t = cell.text.strip()
                if t and t not in cells:
                    cells.append(t)
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def parse_document(filename: str, data: bytes) -> ParsedDocument:
    """Extract plain text from an uploaded file.

    Raises :class:`DocumentParseError` for unsupported types, empty/oversized/corrupt files and
    documents without extractable text (e.g. scanned images).
    """
    if not filename:
        raise DocumentParseError("No file name was provided.")
    ext = Path(filename).suffix.lower()
    if ext == ".doc":
        raise DocumentParseError("Legacy .doc files are not supported. Save the file as .docx or .pdf and upload it again.")
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError(f"Unsupported file type '{ext or 'unknown'}'. Please upload a PDF, DOCX or TXT file.")
    if not data:
        raise DocumentParseError("The uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise DocumentParseError(f"The file is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB. Please upload a smaller file.")

    pages = None
    warnings: list[str] = []
    if ext == ".txt":
        raw = _parse_txt(data)
    elif ext == ".pdf":
        raw, pages = _parse_pdf(data)
    else:
        raw = _parse_docx(data)

    text = normalize_whitespace(raw)
    if not text:
        if ext == ".pdf":
            raise DocumentParseError("No text could be extracted from this PDF. It may be a scanned image; "
                                     "paste the text manually or use a text-based PDF.")
        raise DocumentParseError("No readable text was found in the file.")
    if len(text) < 80:
        warnings.append("Very little text was extracted. Please check the text before analysing.")
    return ParsedDocument(text=text, kind=ext.lstrip("."), filename=filename, pages=pages, warnings=warnings)
