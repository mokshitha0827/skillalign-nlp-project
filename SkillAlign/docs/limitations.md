### Limitations

- **Finite dictionary.** Only the skills listed in `src/skill_dictionary.py` are recognised; unlisted tools and domain skills are invisible to the tool.
- **Single-letter and ambiguous names.** `R`, `C` and `Go` are recognised only through multi-word aliases (e.g. "R programming", "Golang"); a bare "R" is ignored on purpose.
- **Wording matters.** A skill described without its usual name ("trained models") is only Partial at best; unusual spellings may be missed.
- **Related and implied skills are heuristics.** Whether MySQL "implies" SQL or whether Docker is "related" to Kubernetes is a design decision documented in the dictionary, not ground truth.
- **Priority detection is cue-based.** Mixed phrasing can mislabel required vs preferred skills.
- **Similarity is not competence.** Cosine similarity is uncalibrated, differs between embedding and TF-IDF modes, and can be high for superficially similar text. It can raise a skill to Partial only.
- **TF-IDF fallback is lexical.** It cannot capture synonyms the way embeddings can.
- **Extraction quality.** PDF layouts (columns, tables, scanned images) can degrade text; always review the extracted text.
- **Estimates.** Learning durations assume ~5 hours/week and are approximations; resource links are unverified unless the link check is run.
- **English text only.**

### Responsible use and privacy

- SkillAlign evaluates textual alignment, not a person's overall ability or worth.
- Missing evidence does not prove that the user lacks a skill.
- Similarity scores and coverage are not probabilities of employment or success.
- Resume wording and document quality influence the results.
- The tool must not be the sole basis for hiring, rejection or other consequential decisions.
- Review results and correct extraction errors.
- Resumes contain personal information. Processing happens locally; the app stores nothing on disk unless you download an export. The only network use is the one-time model download and the optional link check.
- The tool does not infer sensitive attributes and does not use protected characteristics (name, gender, age, nationality, etc.) in any scoring step.
