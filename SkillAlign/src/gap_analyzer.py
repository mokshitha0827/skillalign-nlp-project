"""Evidence-based skill gap analysis: explicit mentions + alias normalisation + semantic similarity.

Status rules (all configurable in :class:`MatchConfig`)
-------------------------------------------------------
MATCHED  - the required skill (or an alias) is named in at least one resume sentence, or a mentioned skill
           *implies* it (MySQL -> SQL). Semantic similarity alone can never produce MATCHED.
PARTIAL  - not named directly, but (a) a *related* skill is named, or (b) a resume sentence is semantically
           close (cosine similarity >= threshold). Optionally, a skill that appears only inside a skills list
           can be downgraded to PARTIAL (``skills_list_only_status="partial"``).
MISSING  - none of the above: no supporting evidence in the supplied resume.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from . import __version__
from .semantic_matcher import DEFAULT_MODEL, METHOD_EMBEDDINGS, SemanticMatcher
from .skill_dictionary import CATEGORIES, SKILLS
from .skill_extractor import Requirement, SkillMention, extract_mentions, extract_requirements
from .text_preprocessor import Sentence, preprocess

MATCHED, PARTIAL, MISSING = "Matched", "Partial", "Missing"
STATUSES = [MATCHED, PARTIAL, MISSING]


class InputError(ValueError):
    """Raised for empty or unusable user input."""


@dataclass
class MatchConfig:
    method_preference: str = "auto"          # auto | embeddings | tfidf
    model_name: str = DEFAULT_MODEL
    threshold_embedding: float = 0.45        # illustrative, NOT calibrated
    threshold_tfidf: float = 0.25            # illustrative, NOT calibrated
    use_related_skills: bool = True
    use_implied_skills: bool = True
    skills_list_only_status: str = "matched"  # matched | partial
    max_evidence: int = 3
    partial_weight: float = 0.5
    priority_weights: dict = field(default_factory=lambda: {"required": 1.0, "unspecified": 1.0, "preferred": 1.0})
    list_like_min_skills: int = 4
    min_input_chars: int = 20


@dataclass
class Evidence:
    sentence: str
    kind: str                 # explicit | implied | related | semantic
    similarity: float | None
    section: str = "other"
    via: str | None = None    # the resume skill that provided this evidence (implied / related)
    sentence_index: int = -1


@dataclass
class SkillResult:
    skill: str
    category: str
    priority: str
    status: str
    similarity: float | None          # best sentence similarity for this skill (informational)
    evidence: list[Evidence]
    explanation: str
    jd_context: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    results: list[SkillResult]
    stats: dict
    method: str
    method_note: str
    config: dict
    created_at: str
    extra_resume_skills: list[str]
    resume_skill_count: int
    app_version: str = __version__


def skill_query(skill: str) -> str:
    """Text used as the semantic query for a skill: name + short description (never invented credentials)."""
    return f"{skill}: {SKILLS[skill].description}"


def _is_list_like(sent: Sentence, mention_count: int, cfg: MatchConfig) -> bool:
    return sent.section == "skills" or mention_count >= cfg.list_like_min_skills


def compute_stats(results: list[SkillResult], cfg: MatchConfig) -> dict:
    counts = Counter(r.status for r in results)
    total = len(results)
    stats = {
        "total": total, "matched": counts[MATCHED], "partial": counts[PARTIAL], "missing": counts[MISSING],
        "coverage": None, "coverage_required": None, "required_total": 0,
        "formula": "Coverage = sum(w_p * score) / sum(w_p) * 100, score = 1 (Matched), "
                   f"{cfg.partial_weight} (Partial), 0 (Missing); w_p = priority weight (default 1.0 for all).",
    }
    if total:
        w = cfg.priority_weights
        score = {MATCHED: 1.0, PARTIAL: cfg.partial_weight, MISSING: 0.0}
        num = sum(w.get(r.priority, 1.0) * score[r.status] for r in results)
        den = sum(w.get(r.priority, 1.0) for r in results)
        stats["coverage"] = round(100 * num / den, 1) if den else None
    req = [r for r in results if r.priority == "required"]
    stats["required_total"] = len(req)
    if req:
        stats["coverage_required"] = round(100 * sum({MATCHED: 1.0, PARTIAL: cfg.partial_weight, MISSING: 0.0}[r.status] for r in req) / len(req), 1)
    return stats


def category_breakdown(results: list[SkillResult]) -> pd.DataFrame:
    rows = []
    for cat in CATEGORIES:
        rs = [r for r in results if r.category == cat]
        if rs:
            c = Counter(r.status for r in rs)
            rows.append({"Category": cat, MATCHED: c[MATCHED], PARTIAL: c[PARTIAL], MISSING: c[MISSING], "Total": len(rs)})
    return pd.DataFrame(rows, columns=["Category", MATCHED, PARTIAL, MISSING, "Total"])


def analyze(resume_text: str, jd_text: str, config: MatchConfig | None = None,
            matcher: SemanticMatcher | None = None) -> AnalysisResult:
    """Run the full pipeline and return an :class:`AnalysisResult` built only from the supplied texts."""
    cfg = config or MatchConfig()
    if not resume_text or len(resume_text.strip()) < cfg.min_input_chars:
        raise InputError("The resume text is empty or too short. Paste or upload your resume first.")
    if not jd_text or len(jd_text.strip()) < cfg.min_input_chars:
        raise InputError("The job description is empty or too short. Paste a job description first.")

    resume_sents = preprocess(resume_text, "resume")
    jd_sents = preprocess(jd_text, "job")
    if not resume_sents or not jd_sents:
        raise InputError("No readable sentences were found in the supplied text.")

    reqs = extract_requirements(jd_sents)
    mentions = extract_mentions(resume_sents)
    sent_by_idx = {s.index: s for s in resume_sents}
    per_sentence_count = Counter()
    for m in mentions:
        per_sentence_count[m.sentence_index] += 1
    distinct_in_sentence: dict[int, set] = {}
    for m in mentions:
        distinct_in_sentence.setdefault(m.sentence_index, set()).add(m.skill)

    # resume_skill -> list of mentions (explicit)
    explicit: dict[str, list[SkillMention]] = {}
    for m in mentions:
        explicit.setdefault(m.skill, []).append(m)

    matcher = matcher or SemanticMatcher(cfg.method_preference, cfg.model_name)
    method = matcher.load()
    threshold = cfg.threshold_embedding if method == METHOD_EMBEDDINGS else cfg.threshold_tfidf

    req_names = list(reqs.keys())
    sim = matcher.similarity([skill_query(s) for s in req_names], [s.text for s in resume_sents]) if req_names else np.zeros((0, 0))
    method = matcher.method  # may have degraded at runtime
    threshold = cfg.threshold_embedding if method == METHOD_EMBEDDINGS else cfg.threshold_tfidf

    def sim_of(row: int, sent_idx: int) -> float:
        return float(sim[row, sent_idx]) if sim.size else 0.0

    results: list[SkillResult] = []
    for row, skill in enumerate(req_names):
        req: Requirement = reqs[skill]
        best_sim = float(sim[row].max()) if sim.size and sim.shape[1] else 0.0
        evidence: list[Evidence] = []

        # 1) explicit mentions of the skill itself
        exp_idx = sorted({m.sentence_index for m in explicit.get(skill, [])})
        for i in sorted(exp_idx, key=lambda i: -sim_of(row, i))[: cfg.max_evidence]:
            evidence.append(Evidence(sent_by_idx[i].text, "explicit", round(sim_of(row, i), 3), sent_by_idx[i].section, sentence_index=i))

        # 2) implied mentions (child skill that implies this one)
        implied: list[Evidence] = []
        if cfg.use_implied_skills and not evidence:
            for child, ms in explicit.items():
                if skill in SKILLS[child].implies:
                    for i in sorted({m.sentence_index for m in ms}):
                        implied.append(Evidence(sent_by_idx[i].text, "implied", round(sim_of(row, i), 3), sent_by_idx[i].section, via=child, sentence_index=i))
            implied.sort(key=lambda e: -(e.similarity or 0))
            evidence = implied[: cfg.max_evidence]

        status, explanation = MISSING, ""
        if evidence:
            list_only = all(_is_list_like(sent_by_idx[e.sentence_index], len(distinct_in_sentence.get(e.sentence_index, ())), cfg)
                            for e in evidence)
            kind = evidence[0].kind
            if kind == "explicit":
                shown = ", ".join(sorted({m.matched_text for m in explicit[skill]}))
                explanation = f"Named directly in the resume ({shown}); {len(exp_idx)} supporting sentence(s) found."
            else:
                vias = ", ".join(sorted({e.via for e in evidence if e.via}))
                explanation = f"Not named directly, but the resume mentions {vias}, which implies this skill."
            status = MATCHED
            if list_only and cfg.skills_list_only_status == "partial":
                status = PARTIAL
                explanation += " It appears only in a skills list without project or experience context, so it is rated Partial."
            elif list_only:
                explanation += " Note: it appears only in a skills list; no project context was found."
        else:
            # 3) related skills -> partial
            related_ev: list[Evidence] = []
            if cfg.use_related_skills:
                for rel in SKILLS[skill].related:
                    for i in sorted({m.sentence_index for m in explicit.get(rel, [])}):
                        related_ev.append(Evidence(sent_by_idx[i].text, "related", round(sim_of(row, i), 3), sent_by_idx[i].section, via=rel, sentence_index=i))
            related_ev.sort(key=lambda e: -(e.similarity or 0))
            seen = set()
            related_ev = [e for e in related_ev if not (e.sentence in seen or seen.add(e.sentence))][: cfg.max_evidence]
            # 4) semantic similarity -> partial (never matched)
            sem_ev: list[Evidence] = []
            if sim.size:
                order = np.argsort(-sim[row])[: cfg.max_evidence]
                for i in order:
                    if sim[row, i] >= threshold and sent_by_idx[int(i)].text not in {e.sentence for e in related_ev}:
                        sem_ev.append(Evidence(sent_by_idx[int(i)].text, "semantic", round(float(sim[row, i]), 3), sent_by_idx[int(i)].section, sentence_index=int(i)))
            evidence = (related_ev + sem_ev)[: cfg.max_evidence]
            if related_ev:
                vias = ", ".join(sorted({e.via for e in related_ev if e.via}))
                explanation = (f"Not named directly. Related skill(s) found: {vias}. This suggests, but does not establish, "
                               f"the requirement.")
                status = PARTIAL
                if sem_ev:
                    explanation += f" A semantically similar sentence (similarity {sem_ev[0].similarity:.2f}) was also found."
            elif sem_ev:
                status = PARTIAL
                explanation = (f"Not named directly. A resume sentence is semantically similar (similarity "
                               f"{sem_ev[0].similarity:.2f} >= threshold {threshold:.2f}). Similarity alone does not prove the skill.")
            else:
                explanation = (f"No supporting evidence was found in the supplied resume "
                               f"(best sentence similarity {best_sim:.2f} < threshold {threshold:.2f}). "
                               f"This does not prove you lack the skill - it may simply not be written down.")
        results.append(SkillResult(skill, req.category, req.priority, status, round(best_sim, 3), evidence, explanation, req.contexts))

    order = {"required": 0, "unspecified": 1, "preferred": 2}
    results.sort(key=lambda r: (order[r.priority], {MISSING: 0, PARTIAL: 1, MATCHED: 2}[r.status], r.category, r.skill))
    extra = sorted(set(explicit) - set(req_names))
    return AnalysisResult(
        results=results, stats=compute_stats(results, cfg), method=method, method_note=matcher.note,
        config=asdict(cfg), created_at=datetime.now().isoformat(timespec="seconds"),
        extra_resume_skills=extra, resume_skill_count=len(explicit),
    )
