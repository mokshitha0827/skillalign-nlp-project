"""Plain-English wording for results (no jargon) used by the dashboard."""
from __future__ import annotations

from .gap_analyzer import MATCHED, MISSING, PARTIAL, AnalysisResult, SkillResult

ICON = {MATCHED: "✅", PARTIAL: "🟡", MISSING: "❌"}
LABEL = {MATCHED: "Found in resume", PARTIAL: "Partly found", MISSING: "Not found"}
IMPORTANCE = {"required": "Must have", "preferred": "Nice to have", "unspecified": "Mentioned"}


def result_label(r: SkillResult) -> str:
    return f"{ICON[r.status]} {LABEL[r.status]}"


def short_evidence(r: SkillResult, width: int = 110) -> str:
    if not r.evidence:
        return "Nothing found in your resume"
    s = r.evidence[0].sentence
    return s if len(s) <= width else s[: width - 1].rstrip() + "…"


def plain_summary(result: AnalysisResult) -> dict:
    """Headline, verdict and next steps in simple English. Never claims more than the evidence shows."""
    s = result.stats
    if s["total"] == 0:
        return {"headline": "No skills from our skill list were found in the job description.",
                "verdict": "Try a job description that names specific skills or tools.", "gaps": [], "next_steps": []}
    headline = (f"Your resume shows evidence for {s['matched']} of the {s['total']} skills in the job description "
                f"({s['partial']} more are partly shown, {s['missing']} are not shown).")
    base = s["coverage_required"] if s["coverage_required"] is not None else s["coverage"]
    scope = "the must-have skills" if s["coverage_required"] is not None else "the skills asked for"
    if base is None:
        verdict = ""
    elif base >= 80:
        verdict = f"Good news: your resume already shows most of {scope}."
    elif base >= 50:
        verdict = f"Fair start: your resume shows about half or more of {scope}, and some are still missing."
    else:
        verdict = f"Many of {scope} are not shown in your resume yet."
    gaps = [r for r in result.results if r.status != MATCHED and r.priority == "required"]
    if not gaps:
        gaps = [r for r in result.results if r.status != MATCHED][:5]
    steps = []
    if s["partial"] + s["missing"]:
        steps.append("For each gap below, ask yourself: *have I really used this?*")
        steps.append("If yes, open **Resume Improvement**, tick it, and add one true sentence about it.")
        steps.append("If no, open **Learning Roadmap** and learn it first. Do not add skills you cannot explain.")
    else:
        steps.append("Every skill is already shown in your resume. Review the evidence to make sure it reads clearly.")
    return {"headline": headline, "verdict": verdict, "gaps": gaps[:6], "next_steps": steps}

LEARN = {"High": "Learn first", "Medium": "Learn next", "Low": "Learn later"}
LEARN_HELP = {"High": "must-have skills your resume does not show",
              "Medium": "must-have skills that are only partly shown, or skills the job mentions",
              "Low": "nice-to-have skills"}
