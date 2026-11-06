import io

import pytest

from src.document_parser import DocumentParseError, parse_document
from src.text_preprocessor import preprocess, split_sentences


def test_txt_utf8_and_cp1252():
    text = "Skills: Python, SQL, caf\u00e9 analytics for data work and reporting duties."
    assert "caf\u00e9" in parse_document("r.txt", text.encode("utf-8")).text
    assert "caf\u00e9" in parse_document("r.txt", text.encode("cp1252")).text


def test_docx_roundtrip():
    from docx import Document
    d = Document()
    d.add_paragraph("Built machine learning models with scikit-learn and Python for churn prediction.")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text = "Skills"
    t.rows[0].cells[1].text = "Docker"
    buf = io.BytesIO()
    d.save(buf)
    out = parse_document("cv.docx", buf.getvalue())
    assert "scikit-learn" in out.text and "Docker" in out.text


def test_pdf_roundtrip():
    import pymupdf as fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Experienced in Python and SQL for analytics projects and dashboards.")
    out = parse_document("cv.pdf", doc.tobytes())
    assert "Python" in out.text and out.pages == 1


def test_blank_pdf_reports_no_text():
    import pymupdf as fitz
    doc = fitz.open()
    doc.new_page()
    with pytest.raises(DocumentParseError, match="No text"):
        parse_document("scan.pdf", doc.tobytes())


@pytest.mark.parametrize("name,data,msg", [
    ("cv.exe", b"abc", "Unsupported"),
    ("cv.doc", b"abc", "Legacy"),
    ("cv.txt", b"", "empty"),
    ("cv.pdf", b"not a pdf at all", "not a valid PDF"),
    ("cv.docx", b"not a zip", "not a valid Word"),
    ("cv.txt", b"   \n  ", "No readable text"),
    ("cv.txt", b"x" * (5 * 1024 * 1024 + 1), "larger"),
    ("", b"abc", "No file name"),
])
def test_invalid_inputs_raise_friendly_errors(name, data, msg):
    with pytest.raises(DocumentParseError, match=msg):
        parse_document(name, data)


def test_technical_terms_survive_preprocessing():
    sents = preprocess("Skills: C++, C#, .NET, CI/CD, Node.js")
    assert len(sents) == 1
    for term in ("C++", "C#", ".NET", "CI/CD", "Node.js"):
        assert term in sents[0].text


def test_sentence_split_protects_dotted_tokens():
    assert split_sentences("Used Node.js 18.2 and .NET. Then wrote C++ code.") == ["Used Node.js 18.2 and .NET.", "Then wrote C++ code."]
    assert split_sentences("C++") == ["C++"]          # regression: one-character bullets were dropped


def test_sections_and_wrapped_lines():
    s = preprocess("SKILLS\nPython, SQL\nEXPERIENCE\n- trained models and\n  evaluated them\n")
    assert s[0].section == "skills"
    assert s[1].section == "experience" and "evaluated them" in s[1].text
