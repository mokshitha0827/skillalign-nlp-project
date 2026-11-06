"""Dictionary-based skill extraction with alias normalisation and priority detection.

Matching logic (defensible, no naive substring search)
------------------------------------------------------
* Every alias becomes a regex with *token boundaries* that are aware of technical symbols, so ``Java`` never
  matches inside ``JavaScript``, ``SQL`` never matches inside ``NoSQL``/``MySQL``, and ``C++``/``C#``/``.NET`` work.
* Spaces and hyphens inside an alias are interchangeable (``machine-learning`` == ``machine learning``).
* Ambiguous aliases (``!Excel``, ``!Spark``, ``!React`` ...) are case-sensitive.
* When matches overlap, the *longest* wins (``Microsoft SQL Server`` -> SQL Server, not SQL).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .skill_dictionary import SKILLS
from .text_preprocessor import Sentence

_LEFT = r"(?<![A-Za-z0-9_])"
_RIGHT = r"(?![A-Za-z0-9_])"


@dataclass(frozen=True)
class SkillMention:
    skill: str
    matched_text: str
    sentence_index: int
    start: int
    end: int


@dataclass
class Requirement:
    skill: str
    category: str
    priority: str                      # required | preferred | unspecified
    contexts: list[str] = field(default_factory=list)
    mention_count: int = 0


def _alias_regex(alias: str) -> tuple[re.Pattern, str]:
    case_sensitive = alias.startswith("!")
    alias = alias.lstrip("!")
    parts = [re.escape(p) for p in re.split(r"[ \-]+", alias) if p]
    body = r"[\s\-]+".join(parts)
    if alias[-1].isalpha() and not alias.lower().endswith("s") and len(alias) >= 4:
        body += r"(?:s|es)?"   # simple plural handling
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(_LEFT + body + _RIGHT, flags), alias


def _compile_patterns() -> list[tuple[re.Pattern, str, str]]:
    patterns = []
    for entry in SKILLS.values():
        for alias in entry.aliases:
            rx, clean = _alias_regex(alias)
            patterns.append((rx, entry.name, clean))
    return patterns


_PATTERNS = _compile_patterns()


def find_skills_in_text(text: str) -> list[tuple[str, str, int, int]]:
    """Return non-overlapping ``(skill, matched_text, start, end)`` tuples, longest match first-accepted."""
    if not text:
        return []
    candidates = []
    for rx, skill, _alias in _PATTERNS:
        for m in rx.finditer(text):
            candidates.append((skill, m.group(0), m.start(), m.end()))
    candidates.sort(key=lambda c: (-(c[3] - c[2]), c[2]))
    accepted: list[tuple[str, str, int, int]] = []
    for c in candidates:
        if any(c[2] < a[3] and a[2] < c[3] for a in accepted):
            continue
        accepted.append(c)
    accepted.sort(key=lambda c: c[2])
    return accepted


def extract_skill_set(text: str) -> set[str]:
    """Distinct canonical skills mentioned in free text (duplicates and repeated mentions collapse)."""
    return {s for s, *_ in find_skills_in_text(text)}


def extract_mentions(sentences: list[Sentence]) -> list[SkillMention]:
    mentions = []
    for sent in sentences:
        for skill, matched, start, end in find_skills_in_text(sent.text):
            mentions.append(SkillMention(skill, matched, sent.index, start, end))
    return mentions


# ----------------------------------------------------------------------------- priority detection
PREFERRED_CUES = ["not required", "preferred", "nice to have", "nice-to-have", "good to have", "bonus",
                  "a plus", "is a plus", "would be a plus", "desirable", "desired", "ideally", "advantageous",
                  "an advantage", "optional"]
REQUIRED_CUES = ["required", "requires", "requirement", "must have", "must-have", "must", "mandatory", "essential",
                 "minimum", "need to", "needs to", "proven", "proficiency", "proficient"]
_PREF_RX = re.compile(r"(?<![A-Za-z])(?:" + "|".join(re.escape(c) for c in PREFERRED_CUES) + r")(?![A-Za-z])", re.I)
_REQ_RX = re.compile(r"(?<![A-Za-z])(?:" + "|".join(re.escape(c) for c in REQUIRED_CUES) + r")(?![A-Za-z])", re.I)


def _clause_around(text: str, start: int, end: int) -> str:
    left = text.rfind(";", 0, start)
    right = text.find(";", end)
    return text[left + 1: right if right != -1 else len(text)]


def classify_priority(sentence_text: str, section: str, start: int = 0, end: int = 0) -> str:
    """Priority of one skill mention using explicit cues (clause-level), then the section heading."""
    clause = _clause_around(sentence_text, start, end)
    if _PREF_RX.search(clause):
        return "preferred"
    if _REQ_RX.search(clause):
        return "required"
    if section == "preferred":
        return "preferred"
    if section == "required":
        return "required"
    return "unspecified"


def extract_requirements(sentences: list[Sentence]) -> dict[str, Requirement]:
    """Distinct skills requested by a job description with priority and context sentences.

    Mentions inside "about us / benefits" sections are ignored. A skill is ``preferred`` only if *every* mention
    is preferred; any ``required`` mention makes it ``required``.
    """
    reqs: dict[str, Requirement] = {}
    prios: dict[str, list[str]] = {}
    for sent in sentences:
        if sent.section == "ignore":
            continue
        for skill, _m, start, end in find_skills_in_text(sent.text):
            prio = classify_priority(sent.text, sent.section, start, end)
            prios.setdefault(skill, []).append(prio)
            r = reqs.get(skill)
            if r is None:
                r = reqs[skill] = Requirement(skill, SKILLS[skill].category, "unspecified")
            r.mention_count += 1
            if sent.text not in r.contexts and len(r.contexts) < 3:
                r.contexts.append(sent.text)
    for skill, plist in prios.items():
        if "required" in plist:
            reqs[skill].priority = "required"
        elif all(p == "preferred" for p in plist):
            reqs[skill].priority = "preferred"
        else:
            reqs[skill].priority = "unspecified"
    return reqs
