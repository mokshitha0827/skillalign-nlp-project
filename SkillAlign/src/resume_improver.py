"""Honest resume improvement: clean mistakes and add ONLY skills/experience the user confirms.

This module never invents skills, employers, dates, credentials or proficiency. A gap can only be closed by
text that the user wrote or explicitly confirmed. Everything else is formatting and wording hygiene.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

from .gap_analyzer import MATCHED, MISSING, PARTIAL, AnalysisResult
from .skill_dictionary import SOFT as SOFT_CATEGORY
from .skill_extractor import extract_skill_set
from .text_preprocessor import _BULLET_RE, clean_text, detect_heading, preprocess

_PRONOUN = re.compile(r"\b(?:[Ii]|[Mm]y|[Mm]e|[Mm]yself)\b")


@dataclass
class Suggestion:
    skill: str
    status: str
    priority: str
    evidence: str | None
    message: str


def resume_quality_issues(text: str) -> list[str]:
    """Deterministic checks for common resume mistakes. Returns human-readable issues."""
    issues: list[str] = []
    cleaned = clean_text(text)
    if not cleaned:
        return ["The resume is empty."]
    sents = preprocess(cleaned, "resume")
    sections = {s.section for s in sents}
    if "skills" not in sections:
        issues.append("No Skills section was found. Add a 'SKILLS' heading listing the tools you really use.")
    if not sections & {"experience", "projects"}:
        issues.append("No Experience or Projects section was found. Skills are more convincing with project or work context.")
    if "education" not in sections:
        issues.append("No Education section was found.")
    lines = [l for l in cleaned.split("\n") if l.strip()]
    markers = {m.group(0).strip() for l in lines if (m := _BULLET_RE.match(l)) and not m.group(0).strip().rstrip(".)").isdigit()}
    if len(markers) > 1:
        issues.append(f"Inconsistent bullet markers ({', '.join(sorted(markers))}). Use one style.")
    if any(_PRONOUN.search(s.text) for s in sents if s.section in {"experience", "projects"}):
        issues.append("First-person words (I, my, me) appear in experience bullets. Resumes usually start bullets with action verbs.")
    long_ones = [s for s in sents if len(s.text.split()) > 40 and s.section != "skills"]
    if long_ones:
        issues.append(f"{len(long_ones)} sentence(s) are longer than 40 words. Shorter bullets are easier to scan.")
    dup_lines = len(lines) - len({l.strip().lower() for l in lines})
    if dup_lines:
        issues.append(f"{dup_lines} duplicate line(s) found.")
    for s in sents:
        if s.section == "skills":
            items = [i.strip().lower() for i in re.split(r"[,;|]", s.text.split(":", 1)[-1]) if i.strip()]
            if len(items) != len(set(items)):
                issues.append("The Skills section lists the same item more than once.")
                break
    exp = [s for s in sents if s.section in {"experience", "projects"}]
    if exp and not any(re.search(r"\d", s.text) for s in exp):
        issues.append("No numbers or measurable results appear in experience bullets (for example dataset size or accuracy). Add real figures where you have them.")
    return issues


def build_suggestions(result: AnalysisResult) -> list[Suggestion]:
    out = []
    for r in result.results:
        if r.status == MATCHED:
            continue
        if r.status == PARTIAL:
            ev = r.evidence[0].sentence if r.evidence else None
            msg = ("Related evidence exists but the skill is not named. If you really used it, confirm below and name it explicitly.")
        else:
            ev = None
            msg = "No evidence in your resume. Add it only if you truly have this skill; otherwise use the Learning Roadmap."
        out.append(Suggestion(r.skill, r.status, r.priority, ev, msg))
    return out


def clean_resume_text(text: str, stats: dict | None = None) -> str:
    """Fix formatting mistakes without changing meaning: spacing, bullet style, duplicate lines/skills, bullet capitalisation."""
    stats = stats if stats is not None else {}
    for _k in ("bullets_unified", "duplicates_removed", "repeated_skills_removed"):
        stats.setdefault(_k, 0)
    text = clean_text(text)
    out: list[str] = []
    seen_recent: set[str] = set()
    current = "other"
    for raw in text.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            if out and out[-1] != "":
                out.append("")
            continue
        sec, pure = detect_heading(line, "resume")
        if sec is not None:
            current = sec
        is_bullet = bool(_BULLET_RE.match(line)) and not re.match(r"^\s*\d{1,2}[.)]\s", line)
        if is_bullet:
            if not line.lstrip().startswith("- "):
                stats["bullets_unified"] += 1
            body = _BULLET_RE.sub("", line).strip()
            line = "- " + body[:1].upper() + body[1:]
        elif sec is not None and pure:
            line = line.strip().rstrip(":").upper()
        if current == "skills" and not (sec is not None and pure):
            prefix = "- " if line.startswith("- ") else ""
            body = line[2:] if prefix else line
            head, sep, tail = body.partition(":") if ":" in body else ("", "", body)
            items, seen = [], set()
            for item in tail.split(","):
                it = item.strip()
                if it and it.lower() not in seen:
                    seen.add(it.lower())
                    items.append(it)
            stats["repeated_skills_removed"] += max(0, len([i for i in tail.split(",") if i.strip()]) - len(items))
            line = prefix + (head + sep + " " if sep else "") + ", ".join(items)
        key = line.strip().lower()
        if key in seen_recent and not (sec is not None and pure) and len(key) > 12:
            stats["duplicates_removed"] += 1
            continue
        seen_recent.add(key)
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def _add_to_skills_line(lines: list[str], names: list[str]) -> bool:
    for i, line in enumerate(lines):
        sec, pure = detect_heading(line, "resume")
        if sec != "skills":
            continue
        target = i
        if pure:
            target = next((j for j in range(i + 1, len(lines)) if lines[j].strip()), None)
            if target is None or detect_heading(lines[target], "resume")[0] is not None:
                lines.insert(i + 1, ", ".join(names))
                return True
        base = lines[target].rstrip().rstrip(".")
        lines[target] = base + (", " if base.strip() and not base.endswith(":") else " ") + ", ".join(names)
        return True
    return False


def apply_improvements(resume_text: str, confirmed: dict[str, str]) -> tuple[str, list[str]]:
    """Return ``(improved_text, warnings)``.

    ``confirmed`` maps a canonical skill name to an optional sentence written by the user. Only these skills are added.
    """
    warnings: list[str] = []
    text = clean_resume_text(resume_text)
    lines = text.split("\n")
    already = extract_skill_set(text)
    new_names = [s for s in confirmed if s not in already]
    bullets = []
    for skill, sentence in confirmed.items():
        sentence = (sentence or "").strip()
        if not sentence:
            continue
        if skill not in extract_skill_set(sentence):
            warnings.append(f"Your sentence for '{skill}' does not name the skill, so it may not be detected. Mention it by name.")
        bullets.append("- " + sentence[:1].upper() + sentence[1:] + ("" if sentence[-1] in ".!?" else "."))
    if new_names:
        if not _add_to_skills_line(lines, new_names):
            idx = next((i for i, l in enumerate(lines) if detect_heading(l, "resume")[0] is not None), len(lines))
            lines[idx:idx] = ["SKILLS", ", ".join(new_names), ""]
    if bullets:
        lines += ["", "ADDITIONAL EXPERIENCE", *bullets]
    return "\n".join(lines).strip() + "\n", warnings


def to_docx_bytes(text: str) -> bytes:
    from docx import Document
    doc = Document()
    for raw in text.split("\n"):
        if not raw.strip():
            continue
        sec, pure = detect_heading(raw, "resume")
        if pure and sec:
            doc.add_heading(raw.strip().rstrip(":").title(), level=2)
        elif raw.startswith("- "):
            doc.add_paragraph(raw[2:], style="List Bullet")
        else:
            doc.add_paragraph(raw)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------
# Full improvement with a visible change log
# ---------------------------------------------------------------------------------------------
@dataclass
class ImproveOutcome:
    text: str
    warnings: list
    changes: list          # plain-English list of what was changed and why
    summary_line: str | None = None


_FIRST_PERSON = re.compile(r"^(- )(?:I|i)\s+(?!am\b|have\b|had\b|'m\b|was\b)([a-z]+)", re.U)


def _find_skills_line(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        sec, pure = detect_heading(line, "resume")
        if sec != "skills":
            continue
        if not pure:
            return i
        j = next((k for k in range(i + 1, len(lines)) if lines[k].strip()), None)
        if j is not None and detect_heading(lines[j], "resume")[0] is None:
            return j
    return None


def _reorder_skills_line(lines: list[str], rank: dict[str, int]) -> int:
    """Put skills that the job description asks for first. Returns how many items moved (0 if nothing changed)."""
    idx = _find_skills_line(lines)
    if idx is None:
        return 0
    line = lines[idx]
    prefix = "- " if line.startswith("- ") else ""
    body = line[len(prefix):]
    head, sep, tail = body.partition(":") if ":" in body else ("", "", body)
    items = [i.strip() for i in tail.split(",") if i.strip()]
    if len(items) < 2:
        return 0

    def key(item: str) -> int:
        found = extract_skill_set(item)
        ranks = [rank[s] for s in found if s in rank]
        return min(ranks) if ranks else 10_000

    new = sorted(items, key=key)            # stable sort keeps the original order inside each group
    if new == items:
        return 0
    moved = sum(1 for a, b in zip(items, new) if a != b)
    lines[idx] = prefix + (head + sep + " " if sep else "") + ", ".join(new)
    return moved


def _summary_line(result: AnalysisResult) -> str | None:
    """A sentence built ONLY from skills that already have explicit evidence in projects/experience."""
    picks = []
    for r in result.results:
        if (r.status == MATCHED and r.category != SOFT_CATEGORY
                and any(e.kind == "explicit" and e.section in {"projects", "experience"} for e in r.evidence)):
            picks.append(r.skill)
        if len(picks) == 4:
            break
    if len(picks) < 2:
        return None
    joined = ", ".join(picks[:-1]) + " and " + picks[-1]
    return f"Hands-on project experience with {joined}."


def improve_resume(resume_text: str, confirmed: dict[str, str], result: AnalysisResult | None = None,
                   add_summary: bool = True) -> ImproveOutcome:
    """Clean mistakes, align the resume with the job, and add ONLY what the user confirmed. Returns a change log."""
    changes: list[str] = []
    warnings: list[str] = []
    stats: dict = {}
    text = clean_resume_text(resume_text, stats)
    if stats["bullets_unified"]:
        changes.append(f"🧹 Made {stats['bullets_unified']} bullet(s) use the same '-' style.")
    if stats["duplicates_removed"]:
        changes.append(f"🧹 Removed {stats['duplicates_removed']} duplicate line(s).")
    if stats["repeated_skills_removed"]:
        changes.append(f"🧹 Removed {stats['repeated_skills_removed']} repeated skill(s) from the Skills list.")
    lines = text.split("\n")

    fp = 0
    for i, l in enumerate(lines):
        m = _FIRST_PERSON.match(l)
        if m:
            word = m.group(2)
            lines[i] = "- " + word[:1].upper() + word[1:] + l[m.end():]
            fp += 1
    if fp:
        changes.append(f"✏️ Removed 'I' from the start of {fp} bullet(s) (for example 'I built...' became 'Built...').")

    if result is not None:
        group = {"required": 0, "unspecified": 1, "preferred": 2}
        rank = {r.skill: group[r.priority] for r in result.results}
        moved = _reorder_skills_line(lines, rank)
        if moved:
            changes.append("🎯 Moved the skills this job asks for to the front of your Skills list, so they are seen first.")

    already = extract_skill_set("\n".join(lines))
    new_names = [s for s in confirmed if s not in already]
    bullets = []
    for skill, sentence in confirmed.items():
        sentence = (sentence or "").strip()
        if not sentence:
            continue
        if skill not in extract_skill_set(sentence):
            warnings.append(f"Your sentence for '{skill}' does not name the skill, so it may not be detected. Mention it by name.")
        bullets.append("- " + sentence[:1].upper() + sentence[1:] + ("" if sentence[-1] in ".!?" else "."))
    if new_names:
        if not _add_to_skills_line(lines, new_names):
            idx = next((i for i, l in enumerate(lines) if detect_heading(l, "resume")[0] is not None), len(lines))
            lines[idx:idx] = ["SKILLS", ", ".join(new_names), ""]
        changes.append("➕ Added to your Skills list (you confirmed these): " + ", ".join(new_names) + ".")
    if bullets:
        lines += ["", "ADDITIONAL EXPERIENCE", *bullets]
        changes.append(f"➕ Added {len(bullets)} sentence(s) that you wrote, under ADDITIONAL EXPERIENCE.")

    summary = None
    if add_summary and result is not None:
        summary = _summary_line(result)
        if summary and summary.lower() not in "\n".join(lines).lower():
            hi = next((i for i, l in enumerate(lines) if detect_heading(l, "resume") == ("summary", True)), None)
            if hi is not None:
                lines.insert(hi + 1, summary)
            else:
                first = next((i for i, l in enumerate(lines) if detect_heading(l, "resume")[0] is not None), len(lines))
                lines[first:first] = ["SUMMARY", summary, ""]
            changes.append("📝 Added one summary line using only skills your resume already proves in projects or experience.")
        else:
            summary = None
    if not changes:
        changes.append("No changes were needed. Your resume formatting is already clean, and no skills were confirmed.")
    return ImproveOutcome("\n".join(lines).strip() + "\n", warnings, changes, summary)


def diff_lines(before: str, after: str) -> list[tuple[str, str]]:
    """Line-level diff as ``[(tag, line)]`` with tag in ``same | add | del`` (after normalising blank lines)."""
    import difflib
    a = [l for l in clean_text(before).split("\n")]
    b = [l for l in after.strip().split("\n")]
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            out += [("same", l) for l in a[i1:i2]]
        else:
            out += [("del", l) for l in a[i1:i2] if l.strip()]
            out += [("add", l) for l in b[j1:j2] if l.strip()]
    return out
