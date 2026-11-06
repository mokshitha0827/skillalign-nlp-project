import pytest

from src.skill_dictionary import SKILLS, validate_dictionary
from src.skill_extractor import classify_priority, extract_requirements, extract_skill_set, find_skills_in_text
from src.text_preprocessor import preprocess


def test_dictionary_is_consistent():
    assert validate_dictionary() == []
    assert len(SKILLS) > 100


@pytest.mark.parametrize("text,expected", [
    ("I use sklearn daily", {"scikit-learn"}),
    ("scikit learn and scikit-learn", {"scikit-learn"}),
    ("natural language processing (NLP)", {"NLP"}),
    ("Microsoft SQL Server", {"SQL Server"}),
    ("SQL Server and MSSQL", {"SQL Server"}),
    ("Experienced with C++, C# and .NET", {"C++", "C#", ".NET"}),
    ("CI/CD with Node.js", {"CI/CD", "Node.js"}),
    ("Golang services", {"Go"}),
])
def test_alias_normalisation(text, expected):
    assert extract_skill_set(text) == expected


@pytest.mark.parametrize("text,absent", [
    ("I know JavaScript", "Java"),
    ("MongoDB and NoSQL", "SQL"),
    ("MySQL", "SQL"),
    ("I excel at writing", "Excel"),
    ("a spark of joy", "Apache Spark"),
    ("Go to the store", "Go"),
    ("the letter R is a vowel? no", "R"),
    ("react to feedback", "React"),
    ("Javascripting", "JavaScript"),
])
def test_no_false_positives(text, absent):
    assert absent not in extract_skill_set(text)


def test_longest_match_wins():
    hits = find_skills_in_text("Microsoft SQL Server")
    assert [h[0] for h in hits] == ["SQL Server"]


def test_duplicates_and_repeats_collapse():
    text = "Python, python, PYTHON and Python3; machine learning and machine-learning."
    assert extract_skill_set(text) == {"Python", "Machine Learning"}


def test_empty_text():
    assert find_skills_in_text("") == [] and extract_skill_set("   ") == set()


def test_priority_cues_and_sections():
    assert classify_priority("Docker is a plus", "other") == "preferred"
    assert classify_priority("Python is required", "other") == "required"
    assert classify_priority("Kubernetes", "preferred") == "preferred"
    assert classify_priority("SQL", "required") == "required"
    assert classify_priority("Use Spark", "responsibilities") == "unspecified"
    s = "Python required; Docker is a plus"
    assert classify_priority(s, "other", s.index("Docker"), s.index("Docker") + 6) == "preferred"
    assert classify_priority(s, "other", 0, 6) == "required"


def test_requirements_ignore_about_us_and_aggregate():
    jd = preprocess("Requirements\n- SQL\nNice to have\n- Docker\n- SQL\nAbout us\nWe use Terraform.", "job")
    reqs = extract_requirements(jd)
    assert "Terraform" not in reqs
    assert reqs["SQL"].priority == "required" and reqs["SQL"].mention_count == 2
    assert reqs["Docker"].priority == "preferred"
