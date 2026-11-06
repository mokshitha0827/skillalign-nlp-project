"""Text cleanup and sentence segmentation that preserves technical terms.

Design rules
------------
* Never strip punctuation wholesale: ``C++``, ``C#``, ``.NET``, ``CI/CD`` and ``Node.js`` must survive.
* Keep original sentence text so it can be displayed as evidence.
* Track which *section* (skills, experience, requirements, preferred, ...) each sentence belongs to.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_PLACEHOLDER = "\uE000"

RESUME_SECTIONS = {
    "summary": ["summary", "professional summary", "profile", "objective", "career objective", "about me"],
    "skills": ["skills", "technical skills", "key skills", "core skills", "core competencies", "competencies",
               "technologies", "tools & technologies", "tools and technologies", "technical proficiencies",
               "skills & tools", "skills summary"],
    "experience": ["experience", "work experience", "professional experience", "employment", "employment history",
                   "work history", "internships", "internship experience", "additional experience"],
    "projects": ["projects", "academic projects", "personal projects", "selected projects", "project experience"],
    "education": ["education", "academic background", "academics"],
    "certifications": ["certifications", "certificates", "licenses"],
    "achievements": ["achievements", "awards", "honors", "honours", "accomplishments"],
}

JOB_SECTIONS = {
    "required": ["requirements", "required qualifications", "qualifications", "minimum qualifications",
                 "basic qualifications", "must have", "must haves", "must-have", "required skills",
                 "what you bring", "what we're looking for", "what we are looking for", "what you'll need",
                 "skills required", "who you are", "your profile", "key requirements", "essential skills",
                 "required", "what you need"],
    "preferred": ["preferred qualifications", "preferred skills", "nice to have", "nice to haves", "nice-to-have",
                  "bonus points", "bonus", "good to have", "desirable", "desirable skills", "preferred",
                  "what would make you stand out"],
    "responsibilities": ["responsibilities", "key responsibilities", "what you'll do", "what you will do", "role",
                         "about the role", "duties", "the role", "job description", "job summary", "overview",
                         "day to day", "in this role"],
    "ignore": ["about us", "about the company", "who we are", "benefits", "what we offer", "perks",
               "equal opportunity", "equal opportunity employer", "how to apply", "compensation"],
}

_BULLET_RE = re.compile(r"^\s*(?:[\u2022\u2023\u25E6\u2043\u2219\u25AA\u25CF\u00B7\u25A0\u2013\u2014\u2192*\-+>]|\d{1,2}[.)])\s+")
_TERMINAL = re.compile(r"[.!?:;]\s*$")
_ABBREV = re.compile(r"\b(?:e\.g|i\.e|etc|vs|approx|inc|ltd|dr|mr|mrs|ms|jr|sr|st|cf|co|corp)\.", re.I)
_DOTTED = re.compile(r"(?<![\w])(?:[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+|\.[A-Za-z]{2,})(?![\w])")
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass(frozen=True)
class Sentence:
    text: str
    index: int
    section: str = "other"
    line_index: int = 0


def normalize_whitespace(text: str) -> str:
    """Normalize newlines, tabs, non-breaking and zero-width spaces; collapse runs of spaces/blank lines."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)
    text = text.replace("\u00a0", " ").replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Light cleanup: Unicode normalisation (fixes PDF ligatures), smart punctuation, whitespace.

    Punctuation that carries technical meaning (+ # . / -) is deliberately left untouched.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return normalize_whitespace(text)


def _heading_key(line: str) -> str:
    s = re.sub(r"^[#*_\s]+|[#*_\s]+$", "", line.strip())
    s = _BULLET_RE.sub("", s)
    s = s.rstrip(":").strip().lower()
    s = re.sub(r"[^a-z&' \-]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _build_lookup(mapping: dict[str, list[str]]) -> dict[str, str]:
    return {label: section for section, labels in mapping.items() for label in labels}


_RESUME_LOOKUP = _build_lookup(RESUME_SECTIONS)
_JOB_LOOKUP = _build_lookup(JOB_SECTIONS)


def detect_heading(line: str, doc_type: str = "resume") -> tuple[str | None, bool]:
    """Return ``(section, is_pure_heading)``.

    ``Skills:`` alone is a pure heading; ``Skills: Python, SQL`` is an inline label (section set,
    but the line itself is content and is kept).
    """
    lookup = _RESUME_LOOKUP if doc_type == "resume" else _JOB_LOOKUP
    stripped = line.strip()
    if not stripped:
        return None, False
    key = _heading_key(stripped)
    if key in lookup and len(stripped.split()) <= 6 and not stripped.rstrip(":").endswith("."):
        return lookup[key], True
    m = re.match(r"^[*_#\s]*([A-Za-z&' \-]{3,40}?)[*_]*\s*[:\u2013-]\s*\S", stripped)
    if m:
        label = _heading_key(m.group(1))
        if label in lookup:
            return lookup[label], False
    return None, False


def split_sentences(line: str) -> list[str]:
    """Split one line/paragraph into sentences, protecting ``Node.js``, ``.NET``, ``e.g.``, ``3.5`` etc."""
    if not line or not line.strip():
        return []
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", _PLACEHOLDER), line)
    protected = _DOTTED.sub(lambda m: m.group(0).replace(".", _PLACEHOLDER), protected)
    parts = _SPLIT.split(protected.strip())
    out = []
    for p in parts:
        p = p.replace(_PLACEHOLDER, ".").strip()
        if any(ch.isalnum() for ch in p):
            out.append(p)
    return out


def preprocess(text: str, doc_type: str = "resume") -> list[Sentence]:
    """Clean text and return ordered :class:`Sentence` objects with section labels.

    ``doc_type`` is ``"resume"`` or ``"job"`` and selects the section-heading vocabulary.
    """
    text = clean_text(text)
    if not text:
        return []
    section = "other"
    # (section, text, is_bullet, blank_before)
    blocks: list[list] = []
    blank = True
    for raw in text.split("\n"):
        if not raw.strip():
            blank = True
            continue
        sec, pure = detect_heading(raw, doc_type)
        if sec is not None:
            section = sec
            if pure:
                blank = True
                continue
        is_bullet = bool(_BULLET_RE.match(raw))
        content = _BULLET_RE.sub("", raw).strip()
        can_merge = (
            blocks and not blank and not is_bullet and sec is None
            and blocks[-1][0] == section and not _TERMINAL.search(blocks[-1][1])
            and content[:1].islower()
        )
        if can_merge:
            blocks[-1][1] += " " + content
        else:
            blocks.append([section, content, is_bullet])
        blank = False
    sentences: list[Sentence] = []
    for line_idx, (sec, content, _b) in enumerate(blocks):
        for s in split_sentences(content):
            sentences.append(Sentence(text=s, index=len(sentences), section=sec, line_index=line_idx))
    return sentences
