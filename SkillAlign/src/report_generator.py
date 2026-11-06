"""Export analysis results (CSV, JSON, PDF). Every export is built from the real AnalysisResult."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict
from datetime import datetime
from xml.sax.saxutils import escape

import pandas as pd

from . import APP_NAME, APP_SUBTITLE, __version__
from .gap_analyzer import AnalysisResult, MATCHED, MISSING, PARTIAL
from .learning_recommender import ESTIMATE_NOTE, RoadmapItem

DISCLAIMER = ("SkillAlign evaluates textual alignment between a resume and a job description. It is an educational "
              "decision-support tool, not a hiring system. Missing evidence does not prove a lack of skill, and "
              "similarity scores are not probabilities of success. Do not use it as the sole basis for consequential decisions.")

METHODOLOGY_SUMMARY = [
    "Text is cleaned (whitespace, Unicode) without removing technical symbols such as C++, C#, .NET or CI/CD, then split into sentences.",
    "Skills are extracted with a dictionary of canonical names and aliases using boundary-aware matching and longest-match-wins.",
    "Matched = skill named (or implied by a more specific skill) in the resume. Partial = related skill named or a semantically similar sentence. Missing = no supporting evidence.",
    "Semantic similarity is cosine similarity between sentence embeddings (all-MiniLM-L6-v2) or TF-IDF vectors (fallback); it can raise a skill to Partial only, never to Matched.",
    "Coverage = (matched + 0.5 x partial) / total requirements x 100 by default. It is an illustrative measure, not a validated hiring score.",
]


def result_to_dataframe(result: AnalysisResult) -> pd.DataFrame:
    rows = []
    for r in result.results:
        rows.append({
            "Skill": r.skill, "Category": r.category, "Priority": r.priority, "Status": r.status,
            "Best similarity": r.similarity,
            "Evidence count": len(r.evidence),
            "Evidence": " || ".join(f"[{e.kind}{' via ' + e.via if e.via else ''}] {e.sentence}" for e in r.evidence)
                        or "No supporting evidence found in the supplied resume",
            "Explanation": r.explanation,
        })
    return pd.DataFrame(rows, columns=["Skill", "Category", "Priority", "Status", "Best similarity",
                                       "Evidence count", "Evidence", "Explanation"])


def to_csv_bytes(result: AnalysisResult) -> bytes:
    return result_to_dataframe(result).to_csv(index=False, quoting=csv.QUOTE_MINIMAL).encode("utf-8-sig")


def roadmap_to_csv_bytes(roadmap: list[RoadmapItem]) -> bytes:
    rows = [{
        "Skill": i.skill, "Category": i.category, "Gap status": i.status, "Priority": i.priority,
        "Estimated weeks (approx.)": i.estimated_weeks,
        "Prerequisites": "; ".join(p["skill"] for p in i.prerequisites),
        "Activities": " | ".join(i.activities),
        "Resources (unverified unless checked)": " | ".join(f"{r['title']} - {r['url']}" for r in i.resources),
        "Rationale": i.rationale,
    } for i in roadmap]
    cols = ["Skill", "Category", "Gap status", "Priority", "Estimated weeks (approx.)", "Prerequisites", "Activities",
            "Resources (unverified unless checked)", "Rationale"]
    return pd.DataFrame(rows, columns=cols).to_csv(index=False).encode("utf-8-sig")


def to_json_bytes(result: AnalysisResult, roadmap: list[RoadmapItem] | None = None) -> bytes:
    payload = {
        "application": f"{APP_NAME} - {APP_SUBTITLE}", "version": __version__,
        "generated_at": result.created_at, "method": result.method, "method_note": result.method_note,
        "config": result.config, "statistics": result.stats,
        "results": [asdict(r) for r in result.results],
        "resume_skills_not_in_job_description": result.extra_resume_skills,
        "learning_roadmap": [asdict(i) for i in roadmap] if roadmap is not None else [],
        "roadmap_estimate_note": ESTIMATE_NOTE,
        "disclaimer": DISCLAIMER,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def _safe(text: str) -> str:
    """Escape for ReportLab Paragraph and replace characters Helvetica cannot render."""
    return escape(str(text)).encode("latin-1", "replace").decode("latin-1")


def to_pdf_bytes(result: AnalysisResult, roadmap: list[RoadmapItem] | None = None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    navy, blue, light = colors.HexColor("#0B2545"), colors.HexColor("#1D4ED8"), colors.HexColor("#F1F5F9")
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=navy, fontSize=20, alignment=0)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=navy, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#334155"))
    cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"{APP_NAME} report", author=APP_NAME)
    s = result.stats
    story = [Paragraph(f"{APP_NAME} - {APP_SUBTITLE}", h1),
             Paragraph(f"Analysis date: {_safe(result.created_at.replace('T', ' '))} &nbsp;|&nbsp; Method: {_safe(result.method)} &nbsp;|&nbsp; Version {__version__}", small),
             Paragraph(_safe(result.method_note), small), Spacer(1, 6)]

    cov = "n/a" if s["coverage"] is None else f"{s['coverage']}%"
    covr = "n/a" if s["coverage_required"] is None else f"{s['coverage_required']}%"
    story += [Paragraph("Summary statistics", h2)]
    summary = Table([["Requirements", "Matched", "Partial", "Missing", "Coverage (all)", "Coverage (required)"],
                     [s["total"], s["matched"], s["partial"], s["missing"], cov, covr]], hAlign="LEFT")
    summary.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                 ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                 ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                                 ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1"))]))
    story += [summary, Spacer(1, 4), Paragraph(_safe(s["formula"]) + " Coverage is an illustrative measure, not a validated hiring score.", small)]

    colour = {MATCHED: "#15803D", PARTIAL: "#B45309", MISSING: "#B91C1C"}
    for status, title in [(MATCHED, "Matched skills"), (PARTIAL, "Partially supported skills"), (MISSING, "Missing skills")]:
        rs = [r for r in result.results if r.status == status]
        story.append(Paragraph(f"{title} ({len(rs)})", h2))
        if not rs:
            story.append(Paragraph("None.", body))
            continue
        data = [[Paragraph("<b>Skill</b>", cell), Paragraph("<b>Category / priority</b>", cell), Paragraph("<b>Evidence and explanation</b>", cell)]]
        for r in rs:
            ev = "<br/>".join(f"&bull; <i>{_safe(e.kind)}{(' via ' + _safe(e.via)) if e.via else ''}</i>: {_safe(e.sentence[:220])}" for e in r.evidence) \
                 or "No supporting evidence was found in the supplied resume."
            data.append([Paragraph(f"<b>{_safe(r.skill)}</b>", cell),
                         Paragraph(f"{_safe(r.category)}<br/>{_safe(r.priority)}", cell),
                         Paragraph(f"{ev}<br/><font color='#475569'>{_safe(r.explanation)}</font>", cell)])
        t = Table(data, colWidths=[32 * mm, 36 * mm, 108 * mm], repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), light), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                               ("LINEBEFORE", (0, 1), (0, -1), 2, colors.HexColor(colour[status]))]))
        story.append(t)

    if roadmap:
        story += [PageBreak(), Paragraph("Suggested learning roadmap", h2), Paragraph(_safe(ESTIMATE_NOTE) + " Resource links are not verified unless a link check was run.", small), Spacer(1, 4)]
        for it in roadmap:
            pre = ", ".join(f"{p['skill']} ({p['note']})" for p in it.prerequisites) or "none identified"
            lines = [Paragraph(f"<b>{_safe(it.skill)}</b> - {it.priority} priority - approx. {it.estimated_weeks} week(s)", body),
                     Paragraph(_safe(it.rationale), small), Paragraph(f"Prerequisites: {_safe(pre)}", small),
                     Paragraph("Activities: " + _safe("; ".join(it.activities)), small),
                     Paragraph("Resources: " + _safe("; ".join(f"{r['title']} ({r['url']})" for r in it.resources)), small), Spacer(1, 5)]
            story.append(KeepTogether(lines))

    story += [PageBreak(), Paragraph("Methodology", h2)]
    story += [Paragraph("&bull; " + _safe(m), body) for m in METHODOLOGY_SUMMARY]
    story += [Paragraph("Limitations and responsible use", h2), Paragraph(_safe(DISCLAIMER), body),
              Paragraph("&bull; Resume wording, formatting and extraction quality influence the results.<br/>"
                        "&bull; The skill dictionary is finite; skills outside it are not assessed.<br/>"
                        "&bull; Similarity thresholds are illustrative and not calibrated.<br/>"
                        "&bull; Review the evidence yourself and correct extraction errors.", body)]
    doc.build(story)
    return buf.getvalue()


def default_filename(prefix: str, ext: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}.{ext}"
