"""Small, transparent evaluation on the synthetic labelled set in ``data/eval_cases.json``.

Reports (1) skill-detection precision/recall/F1 (resume and job description) and (2) 3-class status
classification metrics for requirements the system extracted. Results are measured on a tiny, author-labelled
dataset and must not be read as general accuracy.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .gap_analyzer import MATCHED, MISSING, PARTIAL, MatchConfig, analyze
from .semantic_matcher import SemanticMatcher
from .skill_extractor import extract_skill_set

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_cases.json"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "reports" / "evaluation_results.json"
LABELS = [MATCHED, PARTIAL, MISSING]


def load_cases(path: Path = DATA_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["cases"]


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn}


def evaluate(cases: list[dict], config: MatchConfig | None = None, matcher: SemanticMatcher | None = None) -> dict:
    cfg = config or MatchConfig()
    matcher = matcher or SemanticMatcher(cfg.method_preference, cfg.model_name)
    tp_r = fp_r = fn_r = tp_j = fp_j = fn_j = 0
    conf = {g: Counter() for g in LABELS}
    per_case, errors, skipped = [], [], []
    for c in cases:
        res = analyze(c["resume"], c["jd"], cfg, matcher)
        pred_status = {r.skill: r.status for r in res.results}
        pred_resume = extract_skill_set(c["resume"])
        gold_resume, gold_jd = set(c["resume_skills"]), set(c["jd_skills"])
        pred_jd = set(pred_status)
        tp_r += len(pred_resume & gold_resume); fp_r += len(pred_resume - gold_resume); fn_r += len(gold_resume - pred_resume)
        tp_j += len(pred_jd & gold_jd); fp_j += len(pred_jd - gold_jd); fn_j += len(gold_jd - pred_jd)
        correct = 0
        n = 0
        for skill, gold in c["expected"].items():
            if skill not in pred_status:
                skipped.append({"case": c["id"], "skill": skill, "reason": "not extracted from the job description"})
                continue
            pred = pred_status[skill]
            conf[gold][pred] += 1
            n += 1
            correct += pred == gold
            if pred != gold:
                errors.append({"case": c["id"], "skill": skill, "gold": gold, "predicted": pred})
        per_case.append({"case": c["id"], "classified": n, "correct": correct,
                         "resume_fp": sorted(pred_resume - gold_resume), "resume_fn": sorted(gold_resume - pred_resume),
                         "jd_fp": sorted(pred_jd - gold_jd), "jd_fn": sorted(gold_jd - pred_jd)})
    total = sum(sum(v.values()) for v in conf.values())
    correct_total = sum(conf[g][g] for g in LABELS)
    per_class = {}
    for g in LABELS:
        tp = conf[g][g]
        fp = sum(conf[o][g] for o in LABELS if o != g)
        fn = sum(conf[g][o] for o in LABELS if o != g)
        per_class[g] = {**prf(tp, fp, fn), "support": sum(conf[g].values())}
    macro_f1 = round(sum(per_class[g]["f1"] for g in LABELS) / len(LABELS), 3)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": matcher.method, "method_note": matcher.note,
        "thresholds": {"embedding": cfg.threshold_embedding, "tfidf": cfg.threshold_tfidf},
        "n_cases": len(cases), "n_requirement_labels": sum(len(c["expected"]) for c in cases),
        "skill_detection": {"resume": prf(tp_r, fp_r, fn_r), "job_description": prf(tp_j, fp_j, fn_j)},
        "status_classification": {"n_classified": total, "accuracy": round(correct_total / total, 3) if total else 0.0,
                                  "macro_f1": macro_f1, "per_class": per_class,
                                  "confusion_matrix": {g: dict(conf[g]) for g in LABELS}},
        "not_extracted_from_jd": skipped, "misclassified": errors, "per_case": per_case,
        "caveats": ["Tiny synthetic dataset (10 cases) labelled by the project author - single annotator.",
                    "Default thresholds were fixed before evaluation; this is not an independent test.",
                    "Results are not evidence of general accuracy or of superiority over any other system."],
    }


def threshold_sweep(cases: list[dict], values=(0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5), method: str = "tfidf") -> list[dict]:
    """Exploratory only: re-evaluates on the SAME cases, so it cannot be used as unbiased tuning evidence."""
    rows = []
    for v in values:
        cfg = MatchConfig(method_preference=method, threshold_tfidf=v, threshold_embedding=v)
        r = evaluate(cases, cfg)
        rows.append({"threshold": v, "accuracy": r["status_classification"]["accuracy"],
                     "macro_f1": r["status_classification"]["macro_f1"]})
    return rows


def run_and_save(method: str = "auto", out: Path = RESULTS_PATH, sweep: bool = True) -> dict:
    cases = load_cases()
    cfg = MatchConfig(method_preference=method)
    result = evaluate(cases, cfg)
    if sweep and result["method"] == "tfidf":
        result["exploratory_threshold_sweep_tfidf"] = threshold_sweep(cases)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run the SkillAlign evaluation on the synthetic labelled set.")
    ap.add_argument("--method", default="auto", choices=["auto", "embeddings", "tfidf"])
    ap.add_argument("--out", default=str(RESULTS_PATH))
    a = ap.parse_args()
    res = run_and_save(a.method, Path(a.out))
    print(json.dumps({k: res[k] for k in ("method", "n_cases", "n_requirement_labels", "skill_detection", "status_classification")}, indent=2))
    print("misclassified:", len(res["misclassified"]), "| not extracted:", len(res["not_extracted_from_jd"]))
    print("saved ->", a.out)
