# SkillAlign — Semantic Skill Gap Analyzer

An explainable NLP project that compares a resume with a job description and shows which required skills are
**Matched**, **Partial** or **Missing**, which resume sentences support each finding, and what to learn next.
It is an educational decision-support tool, **not** a hiring system.

**Techniques:** dictionary-based skill extraction with alias normalisation · pretrained Sentence-Transformer
embeddings (`all-MiniLM-L6-v2`) + cosine similarity · TF-IDF fallback · evidence-based classification ·
rule-based learning roadmap. No model training, no MLOps, no cloud services.

## Requirements
Python 3.10+ (developed and tested on Python 3.12). Tested with: streamlit 1.66.1, plotly 7.1.0, pandas 3.0.2,
numpy 2.4.4, scikit-learn 1.8.0, pymupdf 2.0, python-docx 1.2.0, reportlab 4.4.10, pytest 9.1.1.
(`sentence-transformers` was **not** installable in the build sandbox, see "Verification status".)

## Setup and run

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
If script execution is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501.

## Using the dashboard
1. **Overview** – how it works, limits, and a *Start an analysis* button.
2. **Resume & Job Input** – paste text or upload PDF/DOCX/TXT (extracted text is editable), paste a job description,
   or click **Load example data** (`data/sample_*.txt`). **Clear inputs** / **Reset everything** clear the state.
3. **Skill Match Analysis** – KPI cards, coverage gauge, status donut, category chart, a **skill cloud** (bigger word = more important in the job, colour = result), filterable table
   (status / category / priority / search) and expandable evidence for every skill.
4. **Resume Improvement** – lists formatting/wording mistakes, then for every Missing or Partial skill you tick *"I have genuinely used this"* (optionally adding one true sentence). It builds a cleaned resume containing **only what you confirmed**, lets you download it (TXT/DOCX) and re-analyzes it with a before/after comparison. It never invents skills, employers or credentials.
5. **Learning Roadmap** – prioritised gaps, prerequisites, activities, approximate weeks, resources, optional link check.
6. **Reports & Evaluation** – download **CSV, JSON, PDF** and roadmap CSV (built from the actual results), a troubleshooting check for the embedding model, and an **Evaluation** tab with the measured test results.
   Methodology and limitations are documented in `docs/` and in the project report (not inside the app). Sidebar *Matching settings* expose thresholds and rules.

## Embedding model download and fallback
The first analysis in *Auto* mode downloads `all-MiniLM-L6-v2` (~90 MB) from Hugging Face and caches it
(`~/.cache/huggingface`). Pre-download with `python -m src.semantic_matcher --download`.
If the download fails (offline, firewall, missing package), SkillAlign switches automatically to the **TF-IDF fallback**
and says so in the UI and in every export. You can also force it via sidebar → *TF-IDF only*.
Use *Reports & Evaluation → Download report → Matching engine status → Retry* after fixing your connection.

## Tests and evaluation
```bash
python -m pytest -q                       # 83 unit tests
python -m src.evaluation --method tfidf   # writes reports/evaluation_results.json
python -m src.evaluation --method auto    # uses embeddings if the model can be loaded
```
See `docs/evaluation.md` for the dataset, labelling procedure, measured results and why they must not be read as real-world accuracy.

## Project layout
```
app.py                    Streamlit dashboard (6 pages)
src/document_parser.py    PDF / DOCX / TXT extraction with friendly errors
src/text_preprocessor.py  cleanup, sections, sentence segmentation (keeps C++, .NET, CI/CD ...)
src/skill_dictionary.py   127 skills, aliases, descriptions, related / implied skills
src/skill_extractor.py    boundary-aware matching, longest-match, priority detection
src/semantic_matcher.py   Sentence-Transformer + cosine similarity, TF-IDF fallback
src/gap_analyzer.py       status rules, coverage, explanations
src/learning_recommender.py  roadmap plans, resources, optional link check
src/resume_improver.py    resume cleanup + confirmed-skill additions (no invented content)
src/report_generator.py   CSV / JSON / PDF exports
src/evaluation.py         precision / recall / F1 on the synthetic labelled set
src/ui.py                 CSS, KPI cards, Plotly charts
tests/  data/  docs/  reports/
```

## Known limitations
Finite dictionary; bare single-letter names (R, C, Go) are not matched; related/implied skill relations are heuristics;
similarity scores are uncalibrated; TF-IDF is lexical only; PDF layout can hurt extraction; English only; week estimates and
resource links are approximate/unverified. Read `docs/limitations.md` and the responsible-use statement before relying on any result.

## Verification status (honest summary)
- **Executed:** 74 pytest tests (pass); evaluation in TF-IDF mode; end-to-end dashboard run with Streamlit's headless `AppTest`
  (all pages, buttons, filters, validation); CSV/JSON/PDF generation.
- **Not executed:** the real `all-MiniLM-L6-v2` model (no access to Hugging Face from the build sandbox) — the embedding code path
  is tested only with an injected stand-in encoder; no browser screenshot; resource URLs were not checked.
