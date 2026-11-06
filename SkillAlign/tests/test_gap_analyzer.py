import pytest

from src.gap_analyzer import (MATCHED, MISSING, PARTIAL, InputError, MatchConfig, analyze, category_breakdown, compute_stats)
from src.semantic_matcher import SemanticMatcher

CFG = MatchConfig(method_preference="tfidf")
RESUME = ("Skills\nPython, MySQL, Matplotlib\nExperience\nBuilt a text classifier with scikit-learn. "
          "Presented findings to my team.")


def run(jd, resume=RESUME, cfg=CFG):
    return {r.skill: r for r in analyze(resume, jd, cfg).results}


def test_matched_partial_missing():
    r = run("Requirements\n- Python\n- PostgreSQL\n- Docker")
    assert r["Python"].status == MATCHED and r["Python"].evidence[0].kind == "explicit"
    assert r["PostgreSQL"].status == PARTIAL and r["PostgreSQL"].evidence[0].via == "MySQL"
    assert r["Docker"].status == MISSING and r["Docker"].evidence == []
    assert "No supporting evidence" in r["Docker"].explanation


def test_implied_skill_is_matched():
    r = run("Requirements\n- SQL\n- Data visualization")
    assert r["SQL"].status == MATCHED and r["SQL"].evidence[0].kind == "implied"
    assert r["Data Visualization"].status == MATCHED


def test_high_similarity_alone_never_matches():
    class Always:
        def encode(self, texts, **k):
            import numpy as np
            return np.ones((len(texts), 3))
    m = SemanticMatcher("auto", loader=lambda n: Always())
    res = analyze(RESUME, "Requirements\n- Kubernetes", MatchConfig(), m)
    r = res.results[0]
    assert res.method == "embeddings" and r.similarity > 0.99
    assert r.status == PARTIAL and r.evidence[0].kind == "semantic"   # never Matched


def test_evidence_is_real_resume_text():
    res = analyze(RESUME, "Requirements\n- Python\n- scikit-learn\n- Docker", CFG)
    for r in res.results:
        for e in r.evidence:
            assert e.sentence in RESUME.replace("\n", " ") or e.sentence in RESUME


def test_skills_list_only_downgrade():
    cfg = MatchConfig(method_preference="tfidf", skills_list_only_status="partial")
    r = run("Requirements\n- Python\n- scikit-learn", cfg=cfg)
    assert r["Python"].status == PARTIAL           # only in the skills list
    assert r["scikit-learn"].status == MATCHED     # appears in an experience sentence


def test_flags_can_disable_related_and_implied():
    cfg = MatchConfig(method_preference="tfidf", use_related_skills=False, use_implied_skills=False, threshold_tfidf=0.99)
    r = run("Requirements\n- SQL\n- PostgreSQL", cfg=cfg)
    assert r["SQL"].status == MISSING and r["PostgreSQL"].status == MISSING


def test_coverage_calculation():
    res = analyze(RESUME, "Requirements\n- Python\n- PostgreSQL\n- Docker\n- Kubernetes", CFG)
    s = res.stats
    assert (s["total"], s["matched"], s["partial"], s["missing"]) == (4, 1, 1, 2)
    assert s["coverage"] == pytest.approx(100 * (1 + 0.5) / 4, abs=0.05)
    assert s["coverage_required"] == s["coverage"]


def test_coverage_priority_weights_and_partial_weight():
    cfg = MatchConfig(method_preference="tfidf", partial_weight=1.0, priority_weights={"required": 2.0, "unspecified": 1.0, "preferred": 1.0})
    res = analyze(RESUME, "Requirements\n- Python\nNice to have\n- Docker", cfg)
    assert res.stats["coverage"] == pytest.approx(100 * 2 / 3, abs=0.05)


def test_zero_requirements_handled():
    res = analyze(RESUME, "We are a friendly company that likes people and enjoys coffee breaks.", CFG)
    assert res.stats["total"] == 0 and res.stats["coverage"] is None and res.results == []
    assert compute_stats([], CFG)["coverage"] is None


@pytest.mark.parametrize("resume,jd", [("", "Requirements\n- Python"), ("short", "Requirements\n- Python"),
                                       (RESUME, ""), (RESUME, "   "), (None, "x")])
def test_empty_inputs_raise_input_error(resume, jd):
    with pytest.raises(InputError):
        analyze(resume, jd, CFG)


def test_duplicate_jd_mentions_count_once():
    res = analyze(RESUME, "Requirements\n- Python\n- Python programming with python\nNice to have\n- Python", CFG)
    assert [r.skill for r in res.results].count("Python") == 1 and res.stats["total"] == 1


def test_priority_ordering_and_category_breakdown():
    res = analyze(RESUME, "Requirements\n- Docker\nNice to have\n- Kubernetes", CFG)
    assert [r.priority for r in res.results] == ["required", "preferred"]
    df = category_breakdown(res.results)
    assert df["Total"].sum() == 2 and list(df.columns)[:2] == ["Category", MATCHED]


def test_method_is_reported():
    res = analyze(RESUME, "Requirements\n- Python", CFG)
    assert res.method == "tfidf" and res.method_note
