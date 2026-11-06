### Evaluation design

**Dataset.** `data/eval_cases.json` holds 10 synthetic resume / job-description pairs (77 labelled requirements) written by the project author. They cover data science, backend, frontend, NLP, analytics, DevOps and career-change profiles, plus deliberately tricky cases (Java vs JavaScript, SQL vs NoSQL, C vs C++, the verb "excel", repeated aliases, the single-letter language "R").

**Labelling procedure.** One annotator (the author) labelled, *before running the system*, (a) which dictionary skills each resume names and each job description requests, and (b) the status of every requested skill using this rubric:

- **Matched**: the skill is named in the resume, or implied by a more specific named tool (e.g. MySQL → SQL).
- **Partial**: a closely related skill or tool is named, or a related activity is described without naming the skill.
- **Missing**: no supporting evidence.

**Metrics.** Precision, recall and F1 for skill detection (micro-averaged over cases); accuracy, per-class precision/recall/F1, macro-F1 and a confusion matrix for status classification (only for requirements the system extracted).

### Measured results (TF-IDF fallback, thresholds fixed in advance)

| Task | Precision | Recall | F1 |
|---|---|---|---|
| Resume skill detection (70 labelled mentions) | 1.000 | 0.986 | 0.993 |
| Job-description skill detection (77 labelled requirements) | 1.000 | 0.987 | 0.993 |

| Status class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Matched | 1.000 | 1.000 | 1.000 | 46 |
| Partial | 1.000 | 0.938 | 0.968 | 16 |
| Missing | 0.933 | 1.000 | 0.966 | 14 |

Status accuracy 0.987 (75 / 76 classified requirements), macro-F1 0.978.

**Errors.** (1) *Project Management* in the career-changer case was labelled Partial (a volunteer team was coordinated) but predicted Missing, because no related skill is named. (2) *R* in "Nice to have: R, A/B testing" is not extracted because single-letter language names are deliberately not matched (documented limitation). (3) *Hugging Face* was missed on the resume side because the longer alias "Hugging Face Transformers" won the overlap and mapped to *Transformer Models*.

**A bug found by this evaluation.** A bullet containing only `C++` was dropped by the sentence splitter (one alphanumeric character); this was fixed and covered by a regression test.

### How to read these numbers

- The dataset is tiny, synthetic and written by the same person who designed the dictionary and the rubric, so the scores are optimistic and work as a **regression check, not an accuracy estimate**. No comparison with any other system was made and no superiority is claimed.
- An exploratory threshold sweep on the same data gave identical results for TF-IDF thresholds 0.15 to 0.50 (accuracy 0.987) and slightly lower at 0.10 (0.974). This insensitivity arises because every Partial case in the set is produced by the *related-skill* rule, so the **semantic-similarity component is essentially untested by this dataset**.
- **Embedding mode was not evaluated.** The sandbox used to build the project could not download `all-MiniLM-L6-v2`. The embedding code path is covered by unit tests with an injected stand-in encoder, which verifies the logic, not the model quality. Run `python -m src.evaluation --method embeddings` locally to measure it.
- Planned future work (not performed): a larger set with real, anonymised, consented resumes and multiple annotators with inter-annotator agreement.
