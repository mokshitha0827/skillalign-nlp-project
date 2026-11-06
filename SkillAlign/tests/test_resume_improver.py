from src.gap_analyzer import MATCHED, MatchConfig, analyze
from src.resume_improver import (apply_improvements, build_suggestions, clean_resume_text, resume_quality_issues, to_docx_bytes)
from src.skill_extractor import extract_skill_set

CFG = MatchConfig(method_preference="tfidf")
RESUME = "Priya\nskills:\n* python, Python, sql\nEXPERIENCE\n• built a classifier with scikit-learn\n• built a classifier with scikit-learn\n– wrote queries"
JD = "Requirements\n- Python\n- Docker\n- Kubernetes"


def test_quality_issues_detected():
    issues = " ".join(resume_quality_issues(RESUME))
    assert "Inconsistent bullet" in issues and "duplicate" in issues and "more than once" in issues
    assert resume_quality_issues("") == ["The resume is empty."]


def test_cleaning_fixes_format_but_keeps_meaning():
    c = clean_resume_text(RESUME)
    assert c.count("scikit-learn") == 1
    assert "- Python, sql" in c and "*" not in c and "•" not in c
    assert extract_skill_set(c) >= {"Python", "scikit-learn", "SQL"}


def test_only_confirmed_skills_are_added():
    out, warns = apply_improvements(RESUME, {"Docker": "Containerized my Flask app with Docker in a class project."})
    assert "Docker" in extract_skill_set(out)
    assert "Kubernetes" not in extract_skill_set(out) and warns == []
    assert "ADDITIONAL EXPERIENCE" in out


def test_no_confirmation_means_no_new_skills():
    out, _ = apply_improvements(RESUME, {})
    assert extract_skill_set(out) == extract_skill_set(RESUME)
    assert "ADDITIONAL EXPERIENCE" not in out


def test_warning_when_sentence_does_not_name_skill():
    _, warns = apply_improvements(RESUME, {"Docker": "Deployed things."})
    assert warns and "Docker" in warns[0]


def test_skill_without_sentence_goes_to_skills_line_and_reanalysis_improves():
    before = analyze(RESUME, JD, CFG)
    out, _ = apply_improvements(RESUME, {"Docker": ""})
    after = analyze(out, JD, CFG)
    st = {r.skill: r.status for r in after.results}
    assert st["Docker"] == MATCHED and st["Kubernetes"] != MATCHED
    assert after.stats["coverage"] > before.stats["coverage"]


def test_skills_section_created_when_missing():
    out, _ = apply_improvements("Priya\nEXPERIENCE\n- Built a classifier with scikit-learn.", {"Docker": ""})
    assert "SKILLS" in out and "Docker" in extract_skill_set(out)


def test_suggestions_only_for_gaps():
    res = analyze(RESUME, JD, CFG)
    names = {s.skill for s in build_suggestions(res)}
    assert names == {r.skill for r in res.results if r.status != MATCHED}


def test_docx_export_is_valid():
    import io
    from docx import Document
    doc = Document(io.BytesIO(to_docx_bytes("SKILLS\n- Python, SQL\nEXPERIENCE\n- Built things")))
    assert any("Python" in p.text for p in doc.paragraphs)


# ---------------------------------------------------------------- improve_resume / explain / diff
from src.explain import ICON, IMPORTANCE, LABEL, plain_summary, short_evidence
from src.resume_improver import diff_lines, improve_resume

RES2 = ("Jordan\nSUMMARY\nStudent.\nSKILLS\nPandas, Docker, Python, NumPy\nPROJECTS\n"
        "- Built a classifier with scikit-learn and pandas for text data.\n- I analysed survey data with Python.")
JD2 = "Requirements\n- Python\n- scikit-learn\n- Kubernetes\nNice to have\n- Docker"


def test_improve_resume_changes_are_listed_and_truthful():
    res = analyze(RES2, JD2, CFG)
    out = improve_resume(RES2, {}, res)
    text = " ".join(out.changes)
    assert "Removed 'I'" in text and "front of your Skills" in text and "summary line" in text
    assert "Kubernetes" not in extract_skill_set(out.text)          # nothing unconfirmed is added
    assert [l for l in out.text.split("\n") if l.startswith("- ")][1].startswith("- Analysed")


def test_skills_reordered_required_first_and_original_order_kept_within_group():
    res = analyze(RES2, JD2, CFG)
    out = improve_resume(RES2, {}, res)
    line = next(l for l in out.text.split("\n") if l.startswith("Python"))
    assert line.split(", ")[:2] == ["Python", "Pandas"] or line.split(", ")[0] == "Python"
    assert line.split(", ")[-1] in {"Docker", "Pandas", "NumPy"}


def test_summary_only_uses_proven_technical_skills():
    res = analyze(RES2, JD2, CFG)
    out = improve_resume(RES2, {}, res)
    assert out.summary_line and "Teamwork" not in out.summary_line
    for sk in extract_skill_set(out.summary_line):
        assert any(r.skill == sk and r.status == MATCHED for r in res.results)
    assert improve_resume(RES2, {}, res, add_summary=False).summary_line is None


def test_improve_with_confirmation_raises_coverage_and_is_visible_in_diff():
    res = analyze(RES2, JD2, CFG)
    out = improve_resume(RES2, {"Kubernetes": "Deployed a demo app on Kubernetes (minikube)."}, res)
    after = analyze(out.text, JD2, CFG)
    assert after.stats["coverage"] > res.stats["coverage"]
    added = [l for t, l in diff_lines(RES2, out.text) if t == "add"]
    assert any("Kubernetes" in l for l in added)


def test_no_change_message_when_resume_is_already_clean_and_nothing_confirmed():
    clean = "Jordan\nSKILLS\nPython\nEXPERIENCE\n- Built a model with Python."
    out = improve_resume(clean, {}, None, add_summary=False)
    assert "No changes were needed" in out.changes[0]


def test_plain_summary_wording():
    res = analyze(RES2, JD2, CFG)
    s = plain_summary(res)
    assert str(res.stats["matched"]) in s["headline"] and s["next_steps"]
    assert all(r.priority == "required" for r in s["gaps"])
    assert plain_summary(analyze(RES2, "We like coffee and friendly people in our office.", CFG))["gaps"] == []
    assert ICON["Matched"] == "✅" and LABEL["Missing"] == "Not found" and IMPORTANCE["required"] == "Must have"
    gone = analyze(RES2, "Requirements\n- Terraform\n- Python", CFG)
    r = next(x for x in gone.results if x.status == "Missing")
    assert r.skill == "Terraform" and short_evidence(r) == "Nothing found in your resume"


# ---------------------------------------------------------------- skill cloud
def test_skill_cloud_layout_has_every_skill_inside_canvas_without_overlap():
    from src import ui
    res = analyze(open("data/sample_resume.txt").read(), open("data/sample_job_description.txt").read(), CFG)
    items = ui.cloud_layout(res.results)
    assert {i["skill"] for i in items} == {r.skill for r in res.results}
    for i in items:
        assert 0 <= i["x"] - i["w"] / 2 and i["x"] + i["w"] / 2 <= 1000 and 0 <= i["y"] - i["h"] / 2 and i["y"] + i["h"] / 2 <= 420
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            p, q = items[a], items[b]
            assert abs(p["x"] - q["x"]) * 2 >= p["w"] + q["w"] or abs(p["y"] - q["y"]) * 2 >= p["h"] + q["h"]


def test_skill_cloud_sizes_follow_importance_and_html_is_escaped():
    from src import ui
    res = analyze("Skills\nPython and Pandas for analysis work", "Requirements\n- Python\nNice to have\n- Docker", CFG)
    sizes = {i["skill"]: i["size"] for i in ui.cloud_layout(res.results)}
    assert sizes["Python"] > sizes["Docker"]
    html = ui.skill_cloud_html(res.results)
    assert "Python" in html and "Found in resume" in html and ui.skill_cloud_html([]) == ""
