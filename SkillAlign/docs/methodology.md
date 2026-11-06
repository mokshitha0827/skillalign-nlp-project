### Pipeline

1. **Input and parsing.** Resume text is pasted or extracted locally from PDF (PyMuPDF), DOCX (python-docx) or TXT. Empty, oversized, corrupted, password-protected, scanned-image and unsupported files produce explicit error messages.
2. **Preprocessing.** Unicode normalisation (fixes PDF ligatures), whitespace cleanup, bullet removal and sentence segmentation. Punctuation is *not* stripped, so `C++`, `C#`, `.NET`, `CI/CD` and `Node.js` survive, and abbreviations or dotted tokens do not trigger false sentence breaks. Every sentence keeps its original text and its section (skills, experience, requirements, preferred ...).
3. **Skill extraction and normalisation.** A dictionary of canonical skills (name, category, aliases, short description, related and implied skills) is matched with boundary-aware regular expressions: `Java` never matches inside `JavaScript`, `SQL` never matches inside `NoSQL`, and ambiguous words (`Excel`, `Spark`, `React`) are case-sensitive. Overlaps resolve to the longest alias (`Microsoft SQL Server` → *SQL Server*). Duplicate and repeated mentions collapse into one distinct skill.
4. **Requirement priority.** Cue words in the clause (`required`, `must`, `preferred`, `nice to have`, `a plus` ...) and the surrounding heading (`Requirements`, `Nice to have` ...) label each job-description skill *required*, *preferred* or *unspecified*. Mentions under "About us" or "Benefits" are ignored.
5. **Semantic similarity.** Each requirement becomes a short query (`name: description`). Queries and resume sentences are embedded with the pretrained `all-MiniLM-L6-v2` Sentence Transformer and compared with **cosine similarity**. If the model cannot be loaded, **TF-IDF vectors** (skill aliases normalised to canonical tokens, light stemming, English stop words removed) are used instead. The method used is always shown.
6. **Evidence-based classification.**

| Status | Rule (in order) |
|---|---|
| **Matched** | The skill or an alias is named in the resume, *or* a more specific skill that implies it is named (MySQL → SQL). Optionally downgraded to Partial if it appears only in a skills list. |
| **Partial** | Not named, but a *related* skill is named, *or* a resume sentence has similarity ≥ threshold. |
| **Missing** | None of the above. The explanation states that no supporting evidence was found. |

Semantic similarity alone can only produce *Partial*, never *Matched*.

7. **Coverage.** `Coverage = Σ(wₚ · score) / Σ(wₚ) × 100`, with score 1 for Matched, 0.5 for Partial (configurable) and 0 for Missing; all priority weights default to 1, giving `(matched + 0.5 × partial) / total × 100`. If no requirements are found, coverage is reported as not available. A required-only variant is also shown. Coverage is illustrative: it is **not** a validated hiring score, and it is different from similarity and from employability.
8. **Learning roadmap.** Missing and Partial skills are mapped to a rule-based plan (prerequisites, activities, approximate weeks at ~5 h/week, official or established resources). Priority: High = required and missing; Medium = required and partial, or unspecified and missing; Low = everything else. Prerequisite gaps are scheduled first. Week estimates are planning approximations only.

### Model and thresholds

| Setting | Default | Notes |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | Local inference; ~90 MB download on first use, then cached. |
| Partial threshold (embeddings) | 0.45 | Illustrative; not calibrated or tuned. |
| Partial threshold (TF-IDF) | 0.25 | Illustrative; fixed before evaluation. |
| Partial weight | 0.5 | Configurable in the sidebar. |

Cosine similarity expresses how close two vectors are. It is not a probability, its scale differs between embedding and TF-IDF methods, and it is not proof of competence.
