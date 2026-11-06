"""Rule-based learning roadmap for skills that are Missing or Partial.

Everything here is a *planning aid*: week estimates are approximate (they assume roughly 5 hours/week of
study and a beginner-to-working level), and completing a plan does not guarantee proficiency or employment.
Resource URLs point to official documentation or established educational platforms but are **not verified**
unless :func:`verify_urls` is run with network access.
"""
from __future__ import annotations

import urllib.request
from dataclasses import asdict, dataclass, field

from .gap_analyzer import AnalysisResult, MATCHED, MISSING, PARTIAL
from .skill_dictionary import (CLOUD, DB, DS, NLP, PL, SE, SOFT, SKILLS, VIZ, WEB)

HOURS_PER_WEEK = 5
ESTIMATE_NOTE = (f"The times are approximate guesses that assume about {HOURS_PER_WEEK} hours of study per week. "
                 "They do not guarantee that you will master the skill or get a job.")


@dataclass(frozen=True)
class Resource:
    title: str
    url: str
    provider: str
    verified: bool = False


@dataclass
class Plan:
    prerequisites: tuple[str, ...]
    weeks: int
    activities: tuple[str, ...]
    resources: tuple[Resource, ...]


def _r(title, url, provider):
    return Resource(title, url, provider)


def _p(prereqs, weeks, activities, resources):
    return Plan(tuple(prereqs), weeks, tuple(activities), tuple(resources))


PLANS: dict[str, Plan] = {
    "Python": _p([], 4, ["Work through the official tutorial", "Write 10 small scripts (file handling, functions, classes)", "Solve beginner exercises weekly"],
                 [_r("The Python Tutorial", "https://docs.python.org/3/tutorial/", "Python Software Foundation")]),
    "Java": _p([], 6, ["Learn syntax, classes and collections", "Build a small console application", "Practise with unit tests"],
               [_r("Learn Java (dev.java)", "https://dev.java/learn/", "Oracle")]),
    "C++": _p([], 8, ["Learn syntax, pointers and the STL", "Re-implement basic data structures", "Build a small command-line project"],
              [_r("LearnCpp", "https://www.learncpp.com/", "learncpp.com")]),
    "C#": _p([], 6, ["Complete the C# tour", "Build a small console app", "Practise LINQ and classes"],
             [_r("A tour of C#", "https://learn.microsoft.com/en-us/dotnet/csharp/tour-of-csharp/", "Microsoft Learn")]),
    "Go": _p([], 4, ["Complete the Tour of Go", "Write a small HTTP service"], [_r("A Tour of Go", "https://go.dev/tour/", "Go project")]),
    "R": _p([], 5, ["Work through data import, tidy and visualise chapters", "Reproduce an analysis from a public dataset"],
            [_r("R for Data Science (2e)", "https://r4ds.hadley.nz/", "Wickham et al.")]),
    "TypeScript": _p(["JavaScript"], 3, ["Convert a small JavaScript project to TypeScript", "Learn types, interfaces and generics"],
                     [_r("TypeScript Handbook", "https://www.typescriptlang.org/docs/", "Microsoft")]),
    "Machine Learning": _p(["Python", "Statistics", "Pandas"], 8, ["Complete an introductory ML course", "Train and evaluate classifiers on 3 public datasets", "Write up one end-to-end mini project"],
                           [_r("Intro to Machine Learning", "https://www.kaggle.com/learn/intro-to-machine-learning", "Kaggle Learn"),
                            _r("scikit-learn User Guide", "https://scikit-learn.org/stable/user_guide.html", "scikit-learn")]),
    "Deep Learning": _p(["Machine Learning", "Python"], 8, ["Learn neural-network basics", "Train a small network in PyTorch or Keras", "Reproduce a published tutorial"],
                        [_r("Intro to Deep Learning", "https://www.kaggle.com/learn/intro-to-deep-learning", "Kaggle Learn"),
                         _r("PyTorch Tutorials", "https://pytorch.org/tutorials/", "PyTorch")]),
    "Statistics": _p([], 6, ["Study descriptive statistics, probability and hypothesis testing", "Apply tests to a real dataset"],
                     [_r("Statistics and probability", "https://www.khanacademy.org/math/statistics-probability", "Khan Academy"),
                      _r("OpenIntro Statistics", "https://www.openintro.org/book/os/", "OpenIntro")]),
    "Pandas": _p(["Python"], 2, ["Complete the 10-minute tour", "Clean and summarise a messy CSV"],
                 [_r("10 minutes to pandas", "https://pandas.pydata.org/docs/user_guide/10min.html", "pandas")]),
    "NumPy": _p(["Python"], 1, ["Read the absolute-beginners guide", "Practise array indexing and broadcasting"],
                [_r("NumPy: the absolute basics for beginners", "https://numpy.org/doc/stable/user/absolute_beginners.html", "NumPy")]),
    "scikit-learn": _p(["Python", "Pandas"], 3, ["Follow the user guide for classification and pipelines", "Compare models with cross-validation"],
                       [_r("scikit-learn User Guide", "https://scikit-learn.org/stable/user_guide.html", "scikit-learn")]),
    "TensorFlow": _p(["Python", "Deep Learning"], 4, ["Complete the beginner tutorials", "Train an image or text classifier"], [_r("TensorFlow Tutorials", "https://www.tensorflow.org/tutorials", "TensorFlow")]),
    "PyTorch": _p(["Python", "Deep Learning"], 4, ["Complete the learn-the-basics series", "Train a small model end to end"], [_r("PyTorch Tutorials", "https://pytorch.org/tutorials/", "PyTorch")]),
    "Keras": _p(["Python"], 2, ["Follow the getting-started guide", "Build a small classifier"], [_r("Keras getting started", "https://keras.io/getting_started/", "Keras")]),
    "NLP": _p(["Python", "Machine Learning"], 6, ["Study tokenisation, TF-IDF and embeddings", "Build a text classifier", "Read selected chapters of an NLP textbook"],
              [_r("Hugging Face NLP Course", "https://huggingface.co/learn/nlp-course", "Hugging Face"),
               _r("Speech and Language Processing (3rd ed. draft)", "https://web.stanford.edu/~jurafsky/slp3/", "Jurafsky & Martin")]),
    "Transformer Models": _p(["NLP", "Deep Learning"], 5, ["Read the Transformers documentation introduction", "Fine-tune a small pretrained model on a classification task"],
                             [_r("Transformers documentation", "https://huggingface.co/docs/transformers/index", "Hugging Face")]),
    "Hugging Face": _p(["Python", "NLP"], 3, ["Use pipelines for common tasks", "Load a dataset and a model from the Hub"], [_r("Hugging Face NLP Course", "https://huggingface.co/learn/nlp-course", "Hugging Face")]),
    "spaCy": _p(["Python", "NLP"], 2, ["Complete spaCy 101", "Build a small entity-extraction pipeline"], [_r("spaCy 101", "https://spacy.io/usage/spacy-101", "spaCy")]),
    "NLTK": _p(["Python"], 3, ["Work through the first chapters of the NLTK book", "Tokenise and tag a small corpus"], [_r("Natural Language Processing with Python", "https://www.nltk.org/book/", "NLTK Project")]),
    "Sentence Embeddings": _p(["NLP", "Python"], 2, ["Run the quickstart", "Build a tiny semantic-search demo"], [_r("Sentence Transformers documentation", "https://www.sbert.net/", "SBERT")]),
    "SQL": _p([], 4, ["Learn SELECT, JOIN, GROUP BY and subqueries", "Practise on a sample database daily"], [_r("Intro to SQL", "https://www.kaggle.com/learn/intro-to-sql", "Kaggle Learn")]),
    "PostgreSQL": _p(["SQL"], 3, ["Install locally and follow the tutorial", "Design a small schema with constraints and indexes"], [_r("PostgreSQL Tutorial", "https://www.postgresql.org/docs/current/tutorial.html", "PostgreSQL")]),
    "MySQL": _p(["SQL"], 2, ["Follow the official tutorial", "Create and query a sample schema"], [_r("MySQL Tutorial", "https://dev.mysql.com/doc/refman/8.0/en/tutorial.html", "Oracle MySQL")]),
    "MongoDB": _p([], 2, ["Complete the getting-started guide", "Model and query a small document collection"], [_r("MongoDB Manual", "https://www.mongodb.com/docs/manual/", "MongoDB")]),
    "Docker": _p(["Linux"], 2, ["Complete the Get Started guide", "Containerise a small Python app", "Write a docker-compose file"], [_r("Docker Get Started", "https://docs.docker.com/get-started/", "Docker")]),
    "Kubernetes": _p(["Docker"], 4, ["Complete the Kubernetes basics tutorial", "Deploy a container to a local cluster"], [_r("Kubernetes Basics", "https://kubernetes.io/docs/tutorials/kubernetes-basics/", "Kubernetes")]),
    "Git": _p([], 1, ["Read the first chapters of Pro Git", "Practise branching and merging in a personal repository"], [_r("Pro Git book", "https://git-scm.com/book/en/v2", "git-scm.com")]),
    "GitHub": _p(["Git"], 1, ["Follow the get-started docs", "Open a pull request in a practice repository"], [_r("GitHub Docs: Get started", "https://docs.github.com/en/get-started", "GitHub")]),
    "Linux": _p([], 3, ["Learn core command-line tools", "Practise file permissions and processes in a virtual machine"], [_r("Linux Journey", "https://linuxjourney.com/", "Linux Journey")]),
    "CI/CD": _p(["Git", "GitHub"], 2, ["Create a workflow that runs tests on each push"], [_r("GitHub Actions documentation", "https://docs.github.com/en/actions", "GitHub")]),
    "AWS": _p(["Linux"], 5, ["Follow the getting-started resources", "Deploy a small app and store data in cloud storage"], [_r("AWS Getting Started", "https://aws.amazon.com/getting-started/", "Amazon Web Services")]),
    "Azure": _p([], 5, ["Complete a fundamentals learning path", "Deploy a small web app"], [_r("Microsoft Learn: Azure training", "https://learn.microsoft.com/en-us/training/azure/", "Microsoft Learn")]),
    "GCP": _p([], 5, ["Complete a fundamentals course", "Deploy a small service"], [_r("Google Cloud training", "https://cloud.google.com/learn/training", "Google Cloud")]),
    "Terraform": _p(["AWS"], 3, ["Complete the introductory tutorials", "Provision a small resource from code"], [_r("Terraform tutorials", "https://developer.hashicorp.com/terraform/tutorials", "HashiCorp")]),
    "Apache Airflow": _p(["Python"], 3, ["Follow the official tutorial", "Schedule a small ETL DAG"], [_r("Airflow tutorial", "https://airflow.apache.org/docs/apache-airflow/stable/tutorial/index.html", "Apache Airflow")]),
    "Apache Spark": _p(["Python", "SQL"], 5, ["Read the quick start", "Process a medium-sized dataset with PySpark"], [_r("Apache Spark documentation", "https://spark.apache.org/docs/latest/", "Apache Spark")]),
    "HTML": _p([], 2, ["Complete the HTML basics", "Build a static page"], [_r("MDN Learn Web Development", "https://developer.mozilla.org/en-US/docs/Learn", "MDN")]),
    "CSS": _p(["HTML"], 2, ["Learn the box model, flexbox and grid", "Style a multi-section page"], [_r("MDN Learn Web Development", "https://developer.mozilla.org/en-US/docs/Learn", "MDN")]),
    "JavaScript": _p(["HTML"], 5, ["Learn language fundamentals and the DOM", "Build a small interactive page"], [_r("The Modern JavaScript Tutorial", "https://javascript.info/", "javascript.info")]),
    "React": _p(["JavaScript"], 4, ["Complete the official tutorial", "Build a small component-based app"], [_r("React: Learn", "https://react.dev/learn", "React")]),
    "Node.js": _p(["JavaScript"], 3, ["Follow the Node.js learn guides", "Build a small HTTP server"], [_r("Learn Node.js", "https://nodejs.org/en/learn", "OpenJS Foundation")]),
    "Flask": _p(["Python"], 2, ["Follow the quickstart", "Build a small CRUD app"], [_r("Flask documentation", "https://flask.palletsprojects.com/", "Pallets")]),
    "Django": _p(["Python"], 4, ["Complete the official tutorial", "Build a small app with models and views"], [_r("Django tutorial", "https://docs.djangoproject.com/en/stable/intro/tutorial01/", "Django")]),
    "FastAPI": _p(["Python"], 2, ["Complete the tutorial", "Build a small typed API"], [_r("FastAPI tutorial", "https://fastapi.tiangolo.com/tutorial/", "FastAPI")]),
    "REST APIs": _p(["Python"], 2, ["Learn HTTP verbs, status codes and resource design", "Build and test a small REST API"], [_r("FastAPI tutorial", "https://fastapi.tiangolo.com/tutorial/", "FastAPI")]),
    "Streamlit": _p(["Python"], 1, ["Read the documentation basics", "Build a small data app"], [_r("Streamlit documentation", "https://docs.streamlit.io/", "Streamlit")]),
    "Data Visualization": _p(["Python"], 2, ["Learn chart-selection principles", "Recreate 5 charts with Matplotlib or Plotly"], [_r("Matplotlib tutorials", "https://matplotlib.org/stable/tutorials/index.html", "Matplotlib"), _r("Plotly for Python", "https://plotly.com/python/", "Plotly")]),
    "Tableau": _p([], 3, ["Complete the free training videos", "Publish a dashboard from a public dataset"], [_r("Tableau training", "https://www.tableau.com/learn/training", "Tableau")]),
    "Power BI": _p([], 3, ["Complete a guided learning path", "Build a report from a public dataset"], [_r("Power BI documentation", "https://learn.microsoft.com/en-us/power-bi/", "Microsoft Learn")]),
    "Excel": _p([], 2, ["Practise formulas, pivot tables and charts"], [_r("Excel help and learning", "https://support.microsoft.com/en-us/excel", "Microsoft")]),
    "Agile": _p([], 1, ["Read the Scrum Guide", "Join or simulate a sprint on a team project"], [_r("The Scrum Guide", "https://scrumguides.org/", "Scrum.org / Scrum Guides")]),
    "Unit Testing": _p(["Python"], 2, ["Write pytest tests for an existing script", "Practise test-driven development on a small kata"], [_r("pytest: getting started", "https://docs.pytest.org/en/stable/getting-started.html", "pytest")]),
    "Data Structures & Algorithms": _p(["Python"], 10, ["Study core structures and complexity", "Solve problems regularly and review solutions"], [_r("Introduction to Algorithms (6.006)", "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/", "MIT OpenCourseWare")]),
    "System Design": _p(["REST APIs"], 6, ["Read a system-design primer", "Sketch designs for 3 well-known services"], [_r("System Design Primer", "https://github.com/donnemartin/system-design-primer", "GitHub (community)")]),
    "Design Patterns": _p(["Object-Oriented Programming"], 3, ["Study the classic patterns", "Refactor a small project using two patterns"], [_r("Design Patterns catalogue", "https://refactoring.guru/design-patterns", "Refactoring.Guru")]),
}

_CATEGORY_DEFAULT: dict[str, tuple[int, tuple[str, ...], tuple[Resource, ...]]] = {
    PL: (6, ("Complete the official tutorial", "Build one small project", "Practise with exercises"), (_r("Kaggle Learn (courses)", "https://www.kaggle.com/learn", "Kaggle"),)),
    DS: (5, ("Complete an introductory course", "Apply it to a public dataset", "Write a short project summary"), (_r("Kaggle Learn (courses)", "https://www.kaggle.com/learn", "Kaggle"),)),
    NLP: (5, ("Read an introduction and tutorial", "Implement a small example on public text data"), (_r("Hugging Face NLP Course", "https://huggingface.co/learn/nlp-course", "Hugging Face"),)),
    DB: (3, ("Read the official documentation", "Practise on a sample database"), (_r("Intro to SQL", "https://www.kaggle.com/learn/intro-to-sql", "Kaggle Learn"),)),
    CLOUD: (3, ("Follow the official getting-started guide", "Build a tiny hands-on demo"), (_r("GitHub Docs: Get started", "https://docs.github.com/en/get-started", "GitHub"),)),
    WEB: (3, ("Follow the official tutorial", "Build a small demo project"), (_r("MDN Learn Web Development", "https://developer.mozilla.org/en-US/docs/Learn", "MDN"),)),
    SE: (3, ("Read an introduction", "Apply it in a personal project"), (_r("pytest: getting started", "https://docs.pytest.org/en/stable/getting-started.html", "pytest"),)),
    VIZ: (2, ("Complete a guided tutorial", "Rebuild a public dashboard"), (_r("Plotly for Python", "https://plotly.com/python/", "Plotly"),)),
    SOFT: (4, ("Practise in a real setting (club, team project, volunteering)", "Ask a peer or mentor for feedback", "Record concrete examples to describe on your resume"), (_r("MindTools skills library", "https://www.mindtools.com/", "MindTools"),)),
}


@dataclass
class RoadmapItem:
    skill: str
    category: str
    status: str
    requirement_priority: str
    priority: str                    # High | Medium | Low
    rationale: str
    prerequisites: list[dict]
    activities: list[str]
    estimated_weeks: int
    resources: list[dict]
    estimate_note: str = ESTIMATE_NOTE


def _priority(req_priority: str, status: str) -> str:
    if req_priority in {"required", "unspecified"} and status == MISSING and req_priority == "required":
        return "High"
    if req_priority == "required" and status == PARTIAL:
        return "Medium"
    if req_priority == "unspecified" and status == MISSING:
        return "Medium"
    return "Low"


def plan_for(skill: str) -> Plan:
    if skill in PLANS:
        return PLANS[skill]
    cat = SKILLS[skill].category
    weeks, acts, res = _CATEGORY_DEFAULT[cat]
    return Plan((), weeks, tuple(f"{a} ({skill})" if i == 0 else a for i, a in enumerate(acts)), res)


def build_roadmap(result: AnalysisResult, max_items: int | None = None) -> list[RoadmapItem]:
    """Create a prioritised roadmap from Missing and Partial skills only (never invents gaps)."""
    gaps = [r for r in result.results if r.status in {MISSING, PARTIAL}]
    demonstrated = {r.skill for r in result.results if r.status == MATCHED}
    gap_names = {r.skill for r in gaps}
    items: dict[str, RoadmapItem] = {}
    for r in gaps:
        plan = plan_for(r.skill)
        prio = _priority(r.priority, r.status)
        weeks = plan.weeks if r.status == MISSING else max(1, round(plan.weeks / 2))
        ask = {"required": "The job says this is a must-have.", "preferred": "The job lists this as nice to have.",
               "unspecified": "The job description mentions this skill."}[r.priority]
        if r.status == MISSING:
            why = f"{ask} Your resume does not mention it."
        else:
            why = f"{ask} Your resume shows something close, but does not clearly show this skill. A short project or course would make it clear."
        prereqs = [{"skill": p, "also_gap": p in gap_names,
                    "note": "you already show this in your resume" if p in demonstrated else ("also not shown in your resume yet, so learn it first" if p in gap_names else "not checked in this job")}
                   for p in plan.prerequisites]
        items[r.skill] = RoadmapItem(r.skill, r.category, r.status, r.priority, prio, why, prereqs,
                                     list(plan.activities), weeks, [asdict(x) for x in plan.resources])
    # order: priority, then put prerequisite gaps before the skills that depend on them
    rank = {"High": 0, "Medium": 1, "Low": 2}
    base = sorted(items.values(), key=lambda i: (rank[i.priority], i.estimated_weeks, i.skill))
    ordered: list[RoadmapItem] = []
    seen: set[str] = set()

    def visit(item: RoadmapItem, stack=()):
        if item.skill in seen or item.skill in stack:
            return
        for p in item.prerequisites:
            if p["skill"] in items:
                visit(items[p["skill"]], stack + (item.skill,))
        seen.add(item.skill)
        ordered.append(item)

    for it in base:
        visit(it)
    return ordered[:max_items] if max_items else ordered


def verify_urls(items: list[RoadmapItem], timeout: float = 6.0) -> dict[str, str]:
    """Optional link check (HTTP HEAD/GET). Returns ``{url: 'ok (200)' | 'failed: ...'}``. Needs network access."""
    results: dict[str, str] = {}
    for item in items:
        for res in item.resources:
            url = res["url"]
            if url in results:
                continue
            try:
                req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "SkillAlign-link-check"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    results[url] = f"ok ({resp.status})"
            except Exception as exc:
                results[url] = f"failed: {type(exc).__name__}"
    return results
