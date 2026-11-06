import csv
import io
import json

from src.gap_analyzer import MatchConfig, analyze
from src.learning_recommender import PLANS, build_roadmap, plan_for
from src.report_generator import roadmap_to_csv_bytes, to_csv_bytes, to_json_bytes, to_pdf_bytes
from src.skill_dictionary import SKILLS

CFG = MatchConfig(method_preference="tfidf")
RESUME = "Skills\nPython, MySQL & C++\nBuilt <b>things</b> with Python & caf\u00e9 data \u2713 daily in a team project."
JD = "Requirements\n- Python\n- Machine learning\n- Docker\nNice to have\n- Kubernetes"


def make():
    res = analyze(RESUME, JD, CFG)
    return res, build_roadmap(res)


def test_csv_uses_actual_results():
    res, _ = make()
    rows = list(csv.DictReader(io.StringIO(to_csv_bytes(res).decode("utf-8-sig"))))
    assert {r["Skill"] for r in rows} == {r.skill for r in res.results}
    assert {r["Status"] for r in rows} == {r.status for r in res.results}


def test_json_contains_stats_roadmap_and_disclaimer():
    res, rm = make()
    data = json.loads(to_json_bytes(res, rm))
    assert data["statistics"]["total"] == res.stats["total"] == len(data["results"])
    assert len(data["learning_roadmap"]) == len(rm) > 0
    assert "not a hiring system" in data["disclaimer"] and data["method"] == "tfidf"


def test_pdf_is_valid_and_handles_special_characters(tmp_path):
    res, rm = make()
    pdf = to_pdf_bytes(res, rm)
    assert pdf.startswith(b"%PDF") and len(pdf) > 2000
    import pymupdf as fitz
    text = "".join(p.get_text() for p in fitz.open(stream=pdf, filetype="pdf"))
    assert "SkillAlign" in text and "Docker" in text and "Methodology" in text


def test_pdf_with_no_requirements_and_no_roadmap():
    res = analyze(RESUME, "We are a friendly company that enjoys coffee and long walks.", CFG)
    assert to_pdf_bytes(res, []).startswith(b"%PDF")
    assert json.loads(to_json_bytes(res, []))["statistics"]["coverage"] is None


def test_roadmap_only_covers_gaps_and_orders_prerequisites():
    res, rm = make()
    gap = {r.skill for r in res.results if r.status != "Matched"}
    assert {i.skill for i in rm} == gap
    names = [i.skill for i in rm]
    for i in rm:
        for p in i.prerequisites:
            if p["skill"] in names:
                assert names.index(p["skill"]) < names.index(i.skill)
        assert i.estimated_weeks >= 1 and i.resources and i.activities and "approximate" in i.estimate_note.lower()


def test_roadmap_priority_rules():
    res, rm = make()
    by = {i.skill: i for i in rm}
    assert by["Docker"].priority == "High" and by["Kubernetes"].priority == "Low"


def test_roadmap_empty_when_no_gaps():
    res = analyze("Skills\nPython and Pandas for analysis work", "Requirements\n- Python", CFG)
    assert build_roadmap(res) == []


def test_every_skill_has_a_plan_and_prereqs_exist():
    for name in SKILLS:
        p = plan_for(name)
        assert p.weeks >= 1 and p.activities and p.resources
    for name, p in PLANS.items():
        assert name in SKILLS
        assert all(q in SKILLS for q in p.prerequisites)
        assert all(r.url.startswith("https://") for r in p.resources)


def test_roadmap_csv_has_rows():
    _, rm = make()
    rows = list(csv.DictReader(io.StringIO(roadmap_to_csv_bytes(rm).decode("utf-8-sig"))))
    assert len(rows) == len(rm)
