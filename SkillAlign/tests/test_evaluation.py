from src.evaluation import evaluate, load_cases, prf
from src.gap_analyzer import MatchConfig


def test_prf_math():
    assert prf(8, 2, 2)["precision"] == 0.8 and prf(0, 0, 0)["f1"] == 0.0


def test_dataset_is_well_formed():
    cases = load_cases()
    assert len(cases) >= 10
    for c in cases:
        assert set(c["expected"].values()) <= {"Matched", "Partial", "Missing"}
        assert sorted(c["expected"]) == sorted(c["jd_skills"])


def test_evaluation_runs_and_is_consistent():
    r = evaluate(load_cases(), MatchConfig(method_preference="tfidf"))
    cm = r["status_classification"]["confusion_matrix"]
    assert sum(sum(v.values()) for v in cm.values()) == r["status_classification"]["n_classified"]
    assert r["method"] == "tfidf" and 0 <= r["status_classification"]["accuracy"] <= 1
