"""SkillAlign - Semantic Skill Gap Analyzer (Streamlit dashboard). Run with: streamlit run app.py"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import APP_NAME, APP_SUBTITLE, __version__
from src.document_parser import DocumentParseError, parse_document
from src.evaluation import RESULTS_PATH, run_and_save
from src.gap_analyzer import MATCHED, MISSING, PARTIAL, AnalysisResult, InputError, MatchConfig, analyze
from src.resume_improver import (build_suggestions, diff_lines, improve_resume, resume_quality_issues, to_docx_bytes)
from src.explain import ICON, IMPORTANCE, LABEL, LEARN, plain_summary, result_label, short_evidence
from src.learning_recommender import ESTIMATE_NOTE, HOURS_PER_WEEK, build_roadmap, verify_urls
from src.report_generator import (DISCLAIMER, default_filename, result_to_dataframe, roadmap_to_csv_bytes,
                                  to_csv_bytes, to_json_bytes, to_pdf_bytes)
from src.semantic_matcher import DEFAULT_MODEL, SemanticMatcher, reset_model_cache
from src.skill_dictionary import CATEGORIES, SKILLS
from src import ui

ROOT = Path(__file__).parent
PAGES = ["Overview", "Resume & Job Input", "Skill Match Analysis", "Resume Improvement", "Learning Roadmap", "Reports & Evaluation"]

st.set_page_config(page_title=f"{APP_NAME} | {APP_SUBTITLE}", page_icon="🎯", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------ state
for key, default in {"nav": PAGES[0], "resume_saved": "", "jd_saved": "", "result": None, "roadmap": None,
                     "uploader_key": 0, "improved_text": None, "improved_result": None, "improve_warnings": [], "improve_changes": [], "last_upload_sig": None, "upload_msg": None, "link_check": None,
                     "eval_result": None}.items():
    st.session_state.setdefault(key, default)


def goto(page: str) -> None:
    st.session_state["nav"] = page


def load_example() -> None:
    r = (ROOT / "data" / "sample_resume.txt").read_text(encoding="utf-8")
    j = (ROOT / "data" / "sample_job_description.txt").read_text(encoding="utf-8")
    st.session_state.update(resume_text=r, jd_text=j, resume_saved=r, jd_saved=j)
    st.session_state["result"] = st.session_state["roadmap"] = None
    st.session_state["upload_msg"] = ("info", "Example resume and job description loaded. Click 'Analyze skill gap' to run the analysis.")


def reset_all() -> None:
    st.session_state.update(resume_text="", jd_text="", resume_saved="", jd_saved="")
    st.session_state.update(result=None, roadmap=None, last_upload_sig=None, upload_msg=None, link_check=None, improved_text=None, improved_result=None, improve_warnings=[])
    for k in [k for k in st.session_state if str(k).startswith(("fix_", "fixtxt_", "improved_widget"))]:
        del st.session_state[k]
    st.session_state["uploader_key"] += 1


def read_doc(name: str) -> str:
    p = ROOT / "docs" / name
    return p.read_text(encoding="utf-8") if p.exists() else f"*{name} not found.*"


SETTING_DEFAULTS = {"set_pref": "Auto (embeddings, TF-IDF fallback)", "set_thr_emb": 0.45, "set_thr_tfidf": 0.25,
                    "set_related": True, "set_implied": True, "set_listpartial": False, "set_pw": 0.5}
for _k, _v in SETTING_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def reset_settings() -> None:
    st.session_state.update(SETTING_DEFAULTS)


def settings_summary(cfg: dict) -> str:
    onoff = lambda b: "ON" if b else "OFF"
    return (f"Related skills {onoff(cfg['use_related_skills'])} · Implied skills {onoff(cfg['use_implied_skills'])} · "
            f"Skills-list-only as Partial {onoff(cfg['skills_list_only_status'] == 'partial')} · Partial weight {cfg['partial_weight']} · "
            f"Thresholds: embeddings {cfg['threshold_embedding']}, TF-IDF {cfg['threshold_tfidf']}")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(f'<div class="sa-brand">🎯 {APP_NAME}</div><div class="sa-brand-sub">{APP_SUBTITLE} · v{__version__}</div>', unsafe_allow_html=True)
    st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
    st.divider()
    with st.expander("Matching settings", expanded=False):
        pref_label = st.selectbox("Similarity method", ["Auto (embeddings, TF-IDF fallback)", "TF-IDF only"], key="set_pref",
                                  help="Auto tries the pretrained Sentence Transformer and falls back to TF-IDF if it cannot be loaded.")
        thr_emb = st.slider("Partial threshold: embeddings", 0.20, 0.80, step=0.01, key="set_thr_emb",
                            help="Illustrative cosine-similarity cut-off, not a calibrated probability.")
        thr_tfidf = st.slider("Partial threshold: TF-IDF", 0.05, 0.60, step=0.01, key="set_thr_tfidf")
        use_related = st.checkbox("Related skills give Partial credit", key="set_related")
        use_implied = st.checkbox("Specific tools imply parent skills (MySQL → SQL)", key="set_implied")
        list_partial = st.checkbox("Skills-list-only mentions count as Partial", key="set_listpartial",
                                   help="Downgrades skills that appear only in a skills list, with no project/experience sentence.")
        partial_w = st.slider("Partial weight in coverage", 0.0, 1.0, step=0.05, key="set_pw")
        st.button("Restore default settings", on_click=reset_settings, width="stretch")
    st.button("Reset everything", on_click=reset_all, width="stretch")
    st.caption("🔒 Documents are processed locally in this app. Nothing is uploaded to an external service.")

config = MatchConfig(
    method_preference="tfidf" if pref_label.startswith("TF-IDF") else "auto",
    threshold_embedding=thr_emb, threshold_tfidf=thr_tfidf, use_related_skills=use_related, use_implied_skills=use_implied,
    skills_list_only_status="partial" if list_partial else "matched", partial_weight=partial_w)

page = st.session_state["nav"]


def need_result() -> AnalysisResult | None:
    res = st.session_state["result"]
    if res is None:
        st.info("No analysis yet. Add a resume and a job description, then click **Analyze skill gap**.")
        st.button("Go to Resume & Job Input", on_click=goto, args=("Resume & Job Input",), type="primary")
    return res


# ------------------------------------------------------------------ 1. Overview
if page == "Overview":
    st.markdown(f'<div class="sa-hero"><h1>{APP_NAME}: see how your resume aligns with a job</h1>'
                f'<p>An explainable NLP tool that compares a resume with a job description, shows the resume sentences that support each '
                f'requirement, highlights gaps, and suggests what to learn next. It is a learning aid, not a hiring system.</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    steps = [("1", "Provide text", "Paste or upload your resume (PDF, DOCX, TXT) and paste a job description. Everything stays on your machine."),
             ("2", "Extract and match", "Skills are found with a dictionary of names and aliases, then checked against resume sentences using explicit mentions plus sentence-embedding similarity."),
             ("3", "Review evidence", "Each requirement is Matched, Partial or Missing, with the supporting sentences and a plain-language reason. Gaps become a learning roadmap.")]
    for col, (n, t, b) in zip((c1, c2, c3), steps):
        col.markdown(f'<div class="sa-card"><div class="sa-step">{n}</div><h4>{t}</h4><p>{b}</p></div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([3, 2])
    with left:
        st.markdown(ui.card("How a status is decided",
                            "<ul><li><b>Matched</b>: the skill (or an alias) is named in the resume, or implied by a more specific tool such as MySQL → SQL.</li>"
                            "<li><b>Partial</b>: a related skill is named, or a resume sentence is semantically close. Similarity alone never gives <i>Matched</i>.</li>"
                            "<li><b>Missing</b>: no supporting evidence in the text you supplied.</li></ul>"), unsafe_allow_html=True)
    with right:
        st.markdown(ui.card("Limitations at a glance",
                            "<ul><li>Missing evidence does not prove a missing skill.</li><li>Similarity scores are not probabilities.</li>"
                            f"<li>Only the {len(SKILLS)} skills in the dictionary are assessed.</li><li>Do not use as the sole basis for decisions.</li></ul>"), unsafe_allow_html=True)
    st.write("")
    b1, b2, _ = st.columns([1.2, 1.2, 3])
    b1.button("Start an analysis", type="primary", on_click=goto, args=("Resume & Job Input",), width="stretch")
    b2.button("Try with example data", on_click=lambda: (load_example(), goto("Resume & Job Input")), width="stretch")

# ------------------------------------------------------------------ 2. Input
elif page == "Resume & Job Input":
    st.title("Resume & Job Input")
    st.caption("Provide both texts. You can edit the extracted resume text before analysing.")
    for _w, _saved in (("resume_text", "resume_saved"), ("jd_text", "jd_saved")):
        if _w not in st.session_state:
            st.session_state[_w] = st.session_state[_saved]
    top = st.columns([1.2, 1.2, 4])
    top[0].button("Load example data", on_click=load_example, width="stretch")
    top[1].button("Clear inputs", on_click=reset_all, width="stretch")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Resume")
        uploaded = st.file_uploader("Upload a PDF, DOCX or TXT resume", type=["pdf", "docx", "txt"], key=f"uploader_{st.session_state['uploader_key']}")
        if uploaded is not None:
            sig = (uploaded.name, uploaded.size)
            if st.session_state["last_upload_sig"] != sig:
                st.session_state["last_upload_sig"] = sig
                try:
                    parsed = parse_document(uploaded.name, uploaded.getvalue())
                    st.session_state["resume_text"] = parsed.text
                    extra = f" from {parsed.pages} page(s)" if parsed.pages else ""
                    st.session_state["upload_msg"] = ("success", f"Extracted {len(parsed.text):,} characters{extra} from {parsed.filename}. Review and edit below.")
                    if parsed.warnings:
                        st.session_state["upload_msg"] = ("warning", " ".join(parsed.warnings))
                except DocumentParseError as exc:
                    st.session_state["upload_msg"] = ("error", str(exc))
        msg = st.session_state["upload_msg"]
        if msg:
            getattr(st, msg[0])(msg[1])
        st.text_area("Resume text (paste or edit)", key="resume_text", height=360, placeholder="Paste your resume text here, or upload a file above.")
    with right:
        st.subheader("Job description")
        st.text_area("Job description text", key="jd_text", height=470, placeholder="Paste the job description here. Headings such as 'Requirements' and 'Nice to have' help priority detection.")

    st.session_state["resume_saved"] = st.session_state["resume_text"]
    st.session_state["jd_saved"] = st.session_state["jd_text"]
    st.write("")
    run = st.button("Analyze skill gap", type="primary")
    if run:
        try:
            with st.spinner("Extracting skills and computing similarity..."):
                res = analyze(st.session_state["resume_saved"], st.session_state["jd_saved"], config)
            st.session_state["result"] = res
            st.session_state["roadmap"] = build_roadmap(res)
            st.session_state["link_check"] = None
            st.session_state.update(improved_text=None, improved_result=None, improve_warnings=[])
            for k in [k for k in st.session_state if str(k).startswith(("fix_", "fixtxt_", "improved_widget"))]:
                del st.session_state[k]
            if res.stats["total"] == 0:
                st.warning("No skills from the dictionary were recognised in the job description, so no requirements could be assessed. "
                           "Try a job description that names specific skills or tools.")
            else:
                st.success(f"Analysis complete using **{res.method}** similarity: {res.stats['total']} requirements found.")
            if res.method != "embeddings" and config.method_preference == "auto":
                st.warning(res.method_note)
            st.button("View results", on_click=goto, args=("Skill Match Analysis",), type="primary")
        except InputError as exc:
            st.error(str(exc))
        except Exception as exc:  # last-resort guard: never crash the dashboard
            st.error(f"The analysis failed unexpectedly ({type(exc).__name__}: {exc}). Please check your text and try again.")

# ------------------------------------------------------------------ 3. Analysis
elif page == "Skill Match Analysis":
    st.title("Skill Match Analysis")
    res = need_result()
    if res is not None:
        s = res.stats
        method_txt = "Smart matching (AI sentence embeddings)" if res.method == "embeddings" else "Basic matching (word overlap, TF-IDF backup)"
        st.caption(f"Matching used: **{method_txt}** · {res.created_at.replace('T', ' ')}")
        summ = plain_summary(res)
        gaps_html = "".join(f"<li><b>{ui.escape(r.skill)}</b> — {LABEL[r.status].lower()} ({IMPORTANCE[r.priority].lower()})</li>" for r in summ["gaps"])
        steps_html = "".join(f"<li>{ui.escape(t).replace('**', '')}</li>" for t in summ["next_steps"])
        st.markdown(
            f'<div class="sa-card"><h4>In plain words</h4><p style="font-size:1.05rem;color:#0F172A">{ui.escape(summ["headline"])}</p>'
            f'<p><b>{ui.escape(summ["verdict"])}</b></p>'
            + (f'<p style="margin-top:.6rem"><b>Skills to look at first:</b></p><ul>{gaps_html}</ul>' if gaps_html else "")
            + f'<p style="margin-top:.6rem"><b>What to do next:</b></p><ol>{steps_html}</ol></div>', unsafe_allow_html=True)
        st.write("")
        cov = "n/a" if s["coverage"] is None else f"{s['coverage']}%"
        cols = st.columns(5)
        cols[0].markdown(ui.kpi("Skills asked for", s["total"], "in the job description", ui.NAVY), unsafe_allow_html=True)
        cols[1].markdown(ui.kpi("✅ Found", s["matched"], "clearly shown in your resume", ui.STATUS_COLOR[MATCHED]), unsafe_allow_html=True)
        cols[2].markdown(ui.kpi("🟡 Partly found", s["partial"], "related, but not named", ui.STATUS_COLOR[PARTIAL]), unsafe_allow_html=True)
        cols[3].markdown(ui.kpi("❌ Not found", s["missing"], "nothing in your resume", ui.STATUS_COLOR[MISSING]), unsafe_allow_html=True)
        cols[4].markdown(ui.kpi("Coverage", cov, "a rough guide, not a hiring score", ui.BLUE), unsafe_allow_html=True)
        st.write("")
        if s["total"]:
            c1, c2, c3 = st.columns([1, 1, 1.3])
            c1.plotly_chart(ui.coverage_gauge(s["coverage"]), width="stretch")
            c2.plotly_chart(ui.status_donut(s), width="stretch")
            c3.plotly_chart(ui.category_chart(res.results), width="stretch")

            st.subheader("Skill cloud")
            st.caption("Every skill the job asks for. Bigger words are more important in the job. Colour shows whether your resume proves the skill.")
            st.markdown(ui.skill_cloud_html(res.results), unsafe_allow_html=True)
            st.subheader("Your skills, one by one")
            f1, f2, f3, f4 = st.columns([2, 2, 1.6, 1.6])
            show = f1.radio("Show", ["All", "✅ Found", "🟡 Partly found", "❌ Not found"], horizontal=True)
            imp = f2.radio("Importance", ["All", "Must have", "Nice to have", "Mentioned"], horizontal=True)
            cats = ["All categories"] + sorted({r.category for r in res.results}, key=CATEGORIES.index)
            cat = f3.selectbox("Category", cats)
            query = f4.text_input("Search a skill", placeholder="e.g. python")
            want_status = {"All": None, "✅ Found": MATCHED, "🟡 Partly found": PARTIAL, "❌ Not found": MISSING}[show]
            want_imp = {"All": None, "Must have": "required", "Nice to have": "preferred", "Mentioned": "unspecified"}[imp]
            shown = [r for r in res.results if (want_status is None or r.status == want_status) and (want_imp is None or r.priority == want_imp)
                     and (cat == "All categories" or r.category == cat) and query.strip().lower() in r.skill.lower()]
            if not shown:
                st.info("No skills match these filters.")
            else:
                df = pd.DataFrame([{"Skill": r.skill, "Result": result_label(r), "Importance": IMPORTANCE[r.priority],
                                    "Category": r.category, "What we found in your resume": short_evidence(r)} for r in shown])
                st.dataframe(df, hide_index=True, width="stretch",
                             column_config={"What we found in your resume": st.column_config.TextColumn(width="large")})
                st.caption(f"Showing {len(shown)} of {len(res.results)} skills. Open a skill below to see the exact sentences and the reason.")
                for r in shown:
                    with st.expander(f"{ICON[r.status]} {r.skill} — {LABEL[r.status]} · {IMPORTANCE[r.priority]}"):
                        st.markdown(f"**Why:** {r.explanation}")
                        if r.jd_context:
                            st.markdown("**The job description says:** “" + r.jd_context[0] + "”")
                        if r.evidence:
                            st.markdown("**Where we found it in your resume:**")
                            for e in r.evidence:
                                how = {"explicit": "named directly", "implied": f"implied by {e.via}", "related": f"related skill: {e.via}",
                                       "semantic": "similar meaning"}[e.kind]
                                st.markdown(f'<div class="sa-evidence">{ui.escape(e.sentence)}<br><span class="sa-muted">{how} · in the {e.section} part</span></div>', unsafe_allow_html=True)
                        else:
                            st.markdown("**Where we found it in your resume:** nowhere. No supporting sentence was found.")
            if res.extra_resume_skills:
                with st.expander(f"Other skills in your resume that this job did not ask for ({len(res.extra_resume_skills)})"):
                    st.write(", ".join(res.extra_resume_skills))
            with st.expander("How the numbers are worked out"):
                st.markdown(f"""
- **Found** counts as 1 point, **Partly found** as {res.config['partial_weight']} point, **Not found** as 0.
- **Coverage** = points ÷ number of skills asked for × 100. It is only a rough guide and not a hiring score.
- Required-only coverage: **{'n/a' if s['coverage_required'] is None else str(s['coverage_required']) + '%'}** over {s['required_total']} must-have skill(s).
- A skill is **Found** only when your resume names it. A similar-sounding sentence can make it at most **Partly found**.
- Settings used: {settings_summary(res.config)}
""")

# ------------------------------------------------------------------ 3b. Resume improvement
elif page == "Resume Improvement":
    st.title("Resume Improvement")
    res = need_result()
    if res is not None:
        st.markdown('<div class="sa-note"><b>Simple rule:</b> this page fixes mistakes and lines your resume up with the job, but it only adds a skill '
                    '<b>if you tell us you really have it</b>. It never makes up skills, jobs, dates or certificates.</div>', unsafe_allow_html=True)
        st.write("")
        st.markdown('<span class="sa-step">1</span> **Mistakes we found**', unsafe_allow_html=True)
        issues = resume_quality_issues(st.session_state["resume_saved"])
        if issues:
            for i in issues:
                st.markdown(f"⚠️ {i}")
            st.caption("We fix spacing, bullets, duplicate lines and repeated skills for you. Other problems you can fix by hand in the final box.")
        else:
            st.success("No common mistakes were found.")

        st.markdown('<span class="sa-step">2</span> **Tell us which skills you really have**', unsafe_allow_html=True)
        sugg = build_suggestions(res)
        confirmed: dict[str, str] = {}

        def gap_card(sg):
            with st.container(border=True):
                a, b = st.columns([1.2, 3])
                a.markdown(f"**{sg.skill}**  \n{ICON[sg.status]} {LABEL[sg.status]}  \n<span class='sa-muted'>{IMPORTANCE[sg.priority]}</span>", unsafe_allow_html=True)
                with b:
                    if sg.evidence:
                        st.caption("Your resume has something close to this:")
                        st.markdown(f'<div class="sa-evidence">{ui.escape(sg.evidence)}</div>', unsafe_allow_html=True)
                    else:
                        st.caption("Your resume does not mention this skill.")
                    if st.checkbox(f"Yes, I have really used {sg.skill}", key=f"fix_{sg.skill}"):
                        confirmed[sg.skill] = st.text_input(f"One true sentence about where you used {sg.skill} (optional)", key=f"fixtxt_{sg.skill}",
                                                           placeholder=f"Example: Used {sg.skill} in my final-year project to ...")

        must = [x for x in sugg if x.priority == "required"]
        nice = [x for x in sugg if x.priority != "required"]
        if not sugg:
            st.success("Every skill asked for is already shown in your resume.")
        if must:
            st.markdown("**Must-have skills you are missing**")
            for sg in must:
                gap_card(sg)
        if nice:
            with st.expander(f"Nice-to-have skills ({len(nice)})", expanded=not must):
                for sg in nice:
                    gap_card(sg)
        preview_summary = improve_resume(st.session_state["resume_saved"], {}, res).summary_line
        add_summary = st.checkbox("Add one summary line made only from skills your resume already proves", value=True,
                                  help=preview_summary or "Not enough proven skills to write a summary line.", disabled=preview_summary is None)
        if preview_summary:
            st.caption(f"Preview: “{preview_summary}”")

        st.markdown('<span class="sa-step">3</span> **Build your improved resume**', unsafe_allow_html=True)
        if st.button("Build improved resume", type="primary"):
            out = improve_resume(st.session_state["resume_saved"], confirmed, res, add_summary=add_summary)
            st.session_state["improved_text"] = st.session_state["improved_widget"] = out.text
            st.session_state["improve_warnings"] = out.warnings
            st.session_state["improve_changes"] = out.changes
            st.session_state["improved_result"] = analyze(out.text, st.session_state["jd_saved"], MatchConfig(**res.config))
            st.toast("Your improved resume is ready. Scroll down to see it.", icon="✅")
        if st.session_state["improved_text"] is not None:
            st.success("✅ Your improved resume is ready. **Scroll down** to see the before and after, what changed, and the text to download.")
            for w in st.session_state["improve_warnings"]:
                st.warning(w)
            imp_res = st.session_state["improved_result"]
            if imp_res is not None:
                b4, af = res.stats, imp_res.stats
                st.subheader("Before and after")
                fmt = lambda v: "n/a" if v is None else f"{v}%"
                m = st.columns(4)
                m[0].metric("✅ Found", af["matched"], af["matched"] - b4["matched"])
                m[1].metric("🟡 Partly found", af["partial"], af["partial"] - b4["partial"], delta_color="off")
                m[2].metric("❌ Not found", af["missing"], af["missing"] - b4["missing"], delta_color="inverse")
                m[3].metric("Coverage", fmt(af["coverage"]), None if af["coverage"] is None or b4["coverage"] is None else round(af["coverage"] - b4["coverage"], 1))
                if af["matched"] == b4["matched"] and af["partial"] == b4["partial"]:
                    st.info("The numbers did not change because no new skills were confirmed. Cleaning and re-ordering make a resume easier to read, "
                            "but only real new evidence can raise the match. Go back to step 2 and tick the skills you truly have.")
                left = [r.skill for r in imp_res.results if r.status != MATCHED]
                if left:
                    st.caption("Still not fully shown: " + ", ".join(left) + ". If you do not have these yet, use the Learning Roadmap.")
                else:
                    st.success("Every skill asked for now has evidence in the text you confirmed.")
            st.subheader("What we changed")
            for c in st.session_state["improve_changes"]:
                st.markdown(f"- {c}")
            tabs = st.tabs(["Changes only", "Your improved resume (editable)", "Your original resume"])
            with tabs[0]:
                rows = [(t, l) for t, l in diff_lines(st.session_state["resume_saved"], st.session_state["improved_text"]) if t != "same"]
                if not rows:
                    st.caption("The text is the same as your original.")
                for t, l in rows:
                    color, bg, mark = ("#166534", "#DCFCE7", "＋ added") if t == "add" else ("#991B1B", "#FEE2E2", "－ removed")
                    st.markdown(f'<div style="background:{bg};color:{color};border-radius:8px;padding:.35rem .7rem;margin:.2rem 0;font-size:.92rem">'
                                f'<b>{mark}</b> &nbsp;{ui.escape(l)}</div>', unsafe_allow_html=True)
            with tabs[1]:
                if "improved_widget" not in st.session_state:
                    st.session_state["improved_widget"] = st.session_state["improved_text"]
                st.text_area("You can edit this before using it", key="improved_widget", height=420, label_visibility="collapsed")
                st.session_state["improved_text"] = st.session_state["improved_widget"]
                d1, d2, d3 = st.columns([1.2, 1.2, 1.6])
                d1.download_button("⬇ Download (TXT)", st.session_state["improved_text"], "improved_resume.txt", "text/plain", width="stretch")
                d2.download_button("⬇ Download (DOCX)", to_docx_bytes(st.session_state["improved_text"]), "improved_resume.docx",
                                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document", width="stretch")
                if d3.button("Re-check my edits", width="stretch"):
                    try:
                        st.session_state["improved_result"] = analyze(st.session_state["improved_text"], st.session_state["jd_saved"], MatchConfig(**res.config))
                        st.rerun()
                    except InputError as exc:
                        st.error(str(exc))
            with tabs[2]:
                st.text_area("Original", value=st.session_state["resume_saved"], height=420, disabled=True, label_visibility="collapsed")

# ------------------------------------------------------------------ 4. Roadmap
elif page == "Learning Roadmap":
    st.title("Learning Roadmap")
    res = need_result()
    if res is not None:
        roadmap = st.session_state["roadmap"] or []
        if not roadmap:
            st.success("Every skill in the job description is already shown in your resume, so there is nothing you need to learn for this job.")
        else:
            total_weeks = sum(i.estimated_weeks for i in roadmap)
            first = [i for i in roadmap if i.priority == "High"]
            lead = (f"Start with <b>{ui.escape(first[0].skill)}</b>. " if first else "")
            st.markdown(f'<div class="sa-card"><h4>In plain words</h4><p style="font-size:1.05rem;color:#0F172A">You have <b>{len(roadmap)} skills</b> to work on. {lead}'
                        f'If you study them one after another, it could take about <b>{total_weeks} weeks</b> at around {HOURS_PER_WEEK} hours a week.</p>'
                        f'<p style="margin-top:.4rem">{ESTIMATE_NOTE}</p></div>', unsafe_allow_html=True)
            st.write("")
            k = st.columns(4)
            k[0].markdown(ui.kpi("Learn first", sum(i.priority == "High" for i in roadmap), "must-have, not in your resume", ui.STATUS_COLOR[MISSING]), unsafe_allow_html=True)
            k[1].markdown(ui.kpi("Learn next", sum(i.priority == "Medium" for i in roadmap), "must-have, partly shown", ui.STATUS_COLOR[PARTIAL]), unsafe_allow_html=True)
            k[2].markdown(ui.kpi("Learn later", sum(i.priority == "Low" for i in roadmap), "nice to have", ui.BLUE), unsafe_allow_html=True)
            k[3].markdown(ui.kpi("About", f"{total_weeks} weeks", "in total, a rough guess", ui.NAVY), unsafe_allow_html=True)
            st.write("")
            st.plotly_chart(ui.roadmap_chart(roadmap), width="stretch")
            pick = st.radio("Show", ["All", "Learn first", "Learn next", "Learn later"], horizontal=True)
            wanted = {"All": None, "Learn first": "High", "Learn next": "Medium", "Learn later": "Low"}[pick]
            week = 0
            for n, it in enumerate(roadmap, 1):
                start, week = week + 1, week + it.estimated_weeks
                if wanted and it.priority != wanted:
                    continue
                span = f"week {start}" if it.estimated_weeks == 1 else f"weeks {start} to {week}"
                with st.expander(f"{n}. {it.skill} — {LEARN[it.priority]} · about {it.estimated_weeks} week(s) · {ICON[it.status]} {LABEL[it.status]}", expanded=(n <= 2)):
                    st.markdown(f"**Why this skill:** {it.rationale}")
                    st.markdown(f"**When:** {span} of your plan (a rough guess).")
                    if it.prerequisites:
                        st.markdown("**Learn these first:**")
                        for q in it.prerequisites:
                            st.markdown(f"- {q['skill']} — {q['note']}")
                    st.markdown("**What to do:**")
                    for i, a in enumerate(it.activities, 1):
                        st.markdown(f"{i}. {a}")
                    st.markdown("**Where to learn:**")
                    for r in it.resources:
                        st.markdown(f"- [{r['title']}]({r['url']}) ({r['provider']})")
            st.write("")
            st.caption("The learning links go to official guides and well-known learning sites. They have not been checked unless you press the button below.")
            if st.button("Check that the links work (needs internet)"):
                with st.spinner("Checking links..."):
                    st.session_state["link_check"] = verify_urls(roadmap)
            if st.session_state["link_check"]:
                ok = sum(v.startswith("ok") for v in st.session_state["link_check"].values())
                st.write(f"{ok} of {len(st.session_state['link_check'])} links worked.")
                st.dataframe(pd.DataFrame(st.session_state["link_check"].items(), columns=["Link", "Result"]), hide_index=True, width="stretch")

# ------------------------------------------------------------------ 5. Download report
else:
    st.title("Reports & Evaluation")
    res = st.session_state["result"]
    tab_dl, tab_ev = st.tabs(["Download report", "Evaluation"])
    with tab_dl:
        if res is None:
            st.info("Run an analysis first. Downloads are created from your real results.")
            st.button("Go to Resume & Job Input", on_click=goto, args=("Resume & Job Input",), key="exp_go")
        else:
            roadmap = st.session_state["roadmap"] or []
            st.markdown('<div class="sa-note">The files below contain your results and the resume sentences used as proof. Keep them private.</div>', unsafe_allow_html=True)
            st.write("")
            c = st.columns(4)
            c[0].download_button("⬇ Results (CSV)", to_csv_bytes(res), default_filename("skillalign_results", "csv"), "text/csv", width="stretch")
            c[1].download_button("⬇ Full report (JSON)", to_json_bytes(res, roadmap), default_filename("skillalign_report", "json"), "application/json", width="stretch")
            try:
                c[2].download_button("⬇ Report (PDF)", to_pdf_bytes(res, roadmap), default_filename("skillalign_report", "pdf"), "application/pdf", width="stretch")
            except Exception as exc:
                c[2].error(f"PDF generation failed: {type(exc).__name__}")
            c[3].download_button("⬇ Roadmap (CSV)", roadmap_to_csv_bytes(roadmap), default_filename("skillalign_roadmap", "csv"), "text/csv", width="stretch", disabled=not roadmap)
            st.write("")
            st.dataframe(result_to_dataframe(res).drop(columns=["Evidence", "Best similarity", "Evidence count"]), hide_index=True, width="stretch")
        st.caption(DISCLAIMER)
        with st.expander("Matching engine status (for troubleshooting)"):
            if res is not None:
                st.write(f"Last analysis used **{res.method}**. {res.method_note}")
            b1, b2, _ = st.columns([1.3, 1.3, 3])
            if b1.button("Check embedding model"):
                with st.spinner("Trying to load the model (the first run may download ~90 MB)..."):
                    info = SemanticMatcher("embeddings", DEFAULT_MODEL).info()
                (st.success if info["method"] == "embeddings" else st.warning)(info["note"])
            if b2.button("Retry after a failure"):
                reset_model_cache()
                st.info("Cleared. Click 'Check embedding model' or run a new analysis.")

    with tab_ev:
        st.markdown(read_doc("evaluation.md"))
        st.divider()
        st.subheader("Measured results")
        ev = st.session_state["eval_result"]
        if ev is None and RESULTS_PATH.exists():
            ev = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        e1, e2, _ = st.columns([1.5, 1.5, 3])
        if e1.button("Re-run (TF-IDF backup)"):
            with st.spinner("Evaluating..."):
                ev = st.session_state["eval_result"] = run_and_save("tfidf")
        if e2.button("Re-run (auto method)"):
            with st.spinner("Evaluating..."):
                ev = st.session_state["eval_result"] = run_and_save("auto")
        if ev:
            st.caption(f"Method: **{ev['method']}** · {ev['n_cases']} test cases · {ev['n_requirement_labels']} labelled skills · {ev['generated_at']}")
            st.warning("These numbers come from a small test set written and labelled by the project author. They show the program works as designed. They do not prove real-world accuracy.")
            sd, sc = ev["skill_detection"], ev["status_classification"]
            m = st.columns(4)
            m[0].metric("Skills found in resumes (F1)", sd["resume"]["f1"], help=f"Precision {sd['resume']['precision']}, recall {sd['resume']['recall']}")
            m[1].metric("Skills found in job text (F1)", sd["job_description"]["f1"], help=f"Precision {sd['job_description']['precision']}, recall {sd['job_description']['recall']}")
            m[2].metric("Status accuracy", sc["accuracy"])
            m[3].metric("Status macro-F1", sc["macro_f1"])
            st.dataframe(pd.DataFrame(sc["per_class"]).T.reset_index().rename(columns={"index": "Class"}), hide_index=True, width="stretch")
            st.markdown("**Confusion matrix** (rows = correct answer, columns = what the program said)")
            st.dataframe(pd.DataFrame(sc["confusion_matrix"]).T.fillna(0).astype(int), width="stretch")
            if ev["misclassified"] or ev["not_extracted_from_jd"]:
                with st.expander("Mistakes the program made"):
                    st.write("Wrong status:", ev["misclassified"] or "none")
                    st.write("Skills not picked up from the job description:", ev["not_extracted_from_jd"] or "none")
        else:
            st.info("No evaluation results yet. Click a re-run button above.")
