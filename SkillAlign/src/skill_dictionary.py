"""Maintainable skill dictionary.

Each skill has a canonical name, a category, aliases, a short natural-language
description (used as the semantic "query" for sentence similarity), and two kinds
of relationships:

* ``related``  - skills that suggest, but do not establish, the skill  -> PARTIAL evidence.
* ``implies``  - parent skills that a mention of this skill also evidences
                 (e.g. MySQL implies SQL)                              -> MATCHED (kind="implied").

Alias syntax: a leading ``!`` marks an alias as *case-sensitive* (used for
ambiguous words such as "Excel", "Spark", "React", "Swift", "Rust", "Ruby").
Ambiguous one-letter names (R, C, Go) are only recognised through multi-word
aliases such as "R programming" or "Golang" to avoid false positives.
To add a skill, append one ``_add(...)`` call below; tests validate integrity.
"""
from __future__ import annotations

from dataclasses import dataclass

PL = "Programming Languages"
DS = "Data Science & ML"
NLP = "NLP"
DB = "Databases"
CLOUD = "Cloud & DevOps Tools"
WEB = "Web Development"
SE = "Software Engineering"
VIZ = "Visualization & Analytics"
SOFT = "Communication & Teamwork"

CATEGORIES = [PL, DS, NLP, DB, CLOUD, WEB, SE, VIZ, SOFT]


@dataclass(frozen=True)
class SkillEntry:
    name: str
    category: str
    aliases: tuple[str, ...]
    description: str
    related: tuple[str, ...] = ()
    implies: tuple[str, ...] = ()


SKILLS: dict[str, SkillEntry] = {}


def _add(name, category, aliases, description, related=(), implies=()):
    if name in SKILLS:
        raise ValueError(f"Duplicate skill: {name}")
    SKILLS[name] = SkillEntry(name, category, tuple(aliases), description, tuple(related), tuple(implies))


# ---------------------------------------------------------------- Programming languages
_add("Python", PL, ["Python", "Python3"], "writing programs and scripts in the Python programming language")
_add("Java", PL, ["Java"], "developing applications in the Java programming language", ["Spring Boot", "Object-Oriented Programming"])
_add("C++", PL, ["C++", "!CPP"], "developing software in the C++ programming language", ["Object-Oriented Programming"])
_add("C#", PL, ["C#", "C Sharp", "CSharp"], "developing applications in the C# programming language", [".NET"])
_add("C", PL, ["C programming", "C language", "ANSI C"], "writing low-level programs in the C language", ["C++"])
_add("TypeScript", PL, ["TypeScript"], "writing statically typed JavaScript code using TypeScript", ["JavaScript"])
_add("Go", PL, ["Golang", "Go language", "Go programming"], "building services in the Go programming language")
_add("R", PL, ["!R programming", "!R language", "RStudio"], "statistical computing and data analysis in the R language", ["Statistics"])
_add("Scala", PL, ["Scala"], "functional and object-oriented programming in Scala", ["Apache Spark", "Java"])
_add("Kotlin", PL, ["Kotlin"], "developing applications in Kotlin", ["Java"])
_add("Rust", PL, ["!Rust"], "systems programming in Rust", ["C++"])
_add("Swift", PL, ["!Swift"], "developing iOS applications in Swift")
_add("PHP", PL, ["PHP"], "server-side web development in PHP", ["HTML"])
_add("Ruby", PL, ["!Ruby", "Ruby on Rails"], "programming in Ruby")
_add("MATLAB", PL, ["MATLAB"], "numerical computing and simulation in MATLAB", ["Statistics", "NumPy"])
_add("Bash/Shell Scripting", PL, ["Bash", "Shell scripting", "Shell script", "PowerShell"], "automating tasks with shell scripts on the command line", ["Linux"])

# ---------------------------------------------------------------- Data science & ML
_add("Machine Learning", DS, ["machine learning", "!ML"], "building, training and evaluating machine learning models for classification, regression or clustering",
     ["scikit-learn", "TensorFlow", "PyTorch", "Keras", "XGBoost", "Deep Learning", "Model Evaluation", "Feature Engineering", "Text Classification", "Computer Vision", "Recommender Systems"])
_add("Deep Learning", DS, ["deep learning", "neural networks", "neural network"], "training deep neural networks such as CNNs, RNNs and transformers",
     ["TensorFlow", "PyTorch", "Keras", "Transformer Models", "Computer Vision"])
_add("Statistics", DS, ["statistics", "statistical analysis", "statistical modeling", "statistical modelling", "hypothesis testing", "regression analysis"],
     "applying statistical methods such as hypothesis testing, probability and regression", ["A/B Testing", "Time Series", "R"])
_add("Pandas", DS, ["Pandas"], "manipulating and analysing tabular data with the pandas library", ["NumPy", "Data Analysis", "Data Cleaning"], implies=["Python"])
_add("NumPy", DS, ["NumPy"], "numerical computing with arrays using NumPy", ["Pandas"], implies=["Python"])
_add("scikit-learn", DS, ["scikit-learn", "sklearn"], "building machine learning pipelines with the scikit-learn library", ["Machine Learning", "XGBoost", "Model Evaluation"], implies=["Python"])
_add("TensorFlow", DS, ["TensorFlow"], "building and training neural networks with TensorFlow", ["Keras", "PyTorch", "Deep Learning"])
_add("PyTorch", DS, ["PyTorch"], "building and training neural networks with PyTorch", ["TensorFlow", "Keras", "Deep Learning"])
_add("Keras", DS, ["Keras"], "building neural networks with the Keras API", ["TensorFlow", "Deep Learning", "PyTorch"])
_add("XGBoost", DS, ["XGBoost", "LightGBM", "gradient boosting"], "training gradient boosted tree models", ["scikit-learn", "Machine Learning"])
_add("Feature Engineering", DS, ["feature engineering", "feature selection"], "creating and selecting input features to improve model performance", ["Machine Learning", "Data Cleaning"])
_add("Data Analysis", DS, ["data analysis", "data analytics", "exploratory data analysis", "!EDA", "analyzing data", "analysing data"],
     "exploring and analysing datasets to find patterns and insights", ["Pandas", "Statistics", "Excel", "SQL", "Data Visualization", "Data Cleaning", "NumPy"])
_add("Data Cleaning", DS, ["data cleaning", "data cleansing", "data wrangling", "data preprocessing", "data preparation", "data munging"],
     "cleaning, transforming and preparing raw data for analysis", ["Pandas", "Data Analysis", "Text Preprocessing", "ETL"])
_add("A/B Testing", DS, ["A/B testing", "AB testing", "split testing"], "designing and analysing controlled experiments to compare variants", ["Statistics"])
_add("Time Series", DS, ["time series", "time-series forecasting", "forecasting"], "analysing and forecasting data indexed by time", ["Statistics", "Machine Learning"])
_add("Computer Vision", DS, ["computer vision", "image classification", "object detection", "OpenCV"], "analysing images and video with machine learning", ["Deep Learning", "Machine Learning"])
_add("Model Evaluation", DS, ["model evaluation", "cross-validation", "precision and recall", "F1-score", "ROC curve", "confusion matrix"], "evaluating models with metrics such as precision, recall, F1 and cross-validation", ["Machine Learning", "scikit-learn"])
_add("Recommender Systems", DS, ["recommender systems", "recommendation systems", "recommendation engine", "collaborative filtering"], "building systems that recommend items to users", ["Machine Learning", "Semantic Search"])
_add("Jupyter", DS, ["Jupyter", "Jupyter Notebook", "JupyterLab", "Google Colab"], "doing interactive data analysis in Jupyter notebooks", ["Python", "Data Analysis"])

# ---------------------------------------------------------------- NLP
_add("NLP", NLP, ["natural language processing", "NLP", "computational linguistics", "text mining"], "processing and analysing human language text with computational methods",
     ["spaCy", "NLTK", "Transformer Models", "Text Classification", "Sentiment Analysis", "Named Entity Recognition", "Sentence Embeddings", "Hugging Face", "Text Preprocessing", "Topic Modeling", "Word2Vec", "TF-IDF", "LLMs", "Semantic Search"])
_add("Transformer Models", NLP, ["transformer models", "transformer architecture", "BERT", "RoBERTa", "DistilBERT", "Hugging Face Transformers"], "using pretrained transformer language models such as BERT",
     ["Hugging Face", "LLMs", "Sentence Embeddings", "Deep Learning", "NLP"])
_add("Hugging Face", NLP, ["Hugging Face", "HuggingFace"], "using the Hugging Face ecosystem of pretrained models and datasets", ["Transformer Models", "LLMs", "Sentence Embeddings"])
_add("spaCy", NLP, ["spaCy"], "building NLP pipelines with the spaCy library", ["NLTK", "NLP", "Named Entity Recognition"], implies=["Python"])
_add("NLTK", NLP, ["NLTK", "Natural Language Toolkit"], "processing text with the NLTK library", ["spaCy", "NLP", "Text Preprocessing"], implies=["Python"])
_add("Sentence Embeddings", NLP, ["sentence embeddings", "sentence transformers", "sentence-transformers", "sentence-BERT", "SBERT", "text embeddings", "word embeddings", "embeddings"],
     "representing sentences as dense vectors with pretrained embedding models", ["Transformer Models", "Hugging Face", "Word2Vec", "Semantic Search"])
_add("Text Classification", NLP, ["text classification", "document classification", "text categorization"], "classifying documents or sentences into categories", ["NLP", "Machine Learning", "Sentiment Analysis"])
_add("Named Entity Recognition", NLP, ["named entity recognition", "!NER", "entity extraction", "information extraction"], "extracting named entities such as people, organisations and skills from text", ["NLP", "spaCy"])
_add("Sentiment Analysis", NLP, ["sentiment analysis", "opinion mining"], "determining the sentiment expressed in text", ["Text Classification", "NLP"])
_add("Topic Modeling", NLP, ["topic modeling", "topic modelling", "!LDA", "latent dirichlet allocation"], "discovering topics in document collections", ["NLP", "Text Classification"])
_add("LLMs", NLP, ["large language models", "large language model", "LLMs", "LLM", "GPT-4", "ChatGPT"], "working with large language models and generative AI", ["Prompt Engineering", "RAG", "Transformer Models", "Hugging Face"])
_add("RAG", NLP, ["retrieval-augmented generation", "retrieval augmented generation", "!RAG"], "building retrieval-augmented generation systems that ground LLM answers in documents", ["LLMs", "Semantic Search", "Sentence Embeddings"])
_add("TF-IDF", NLP, ["TF-IDF", "TFIDF", "term frequency-inverse document frequency", "term frequency inverse document frequency"], "representing text with term frequency inverse document frequency weights", ["NLP", "Text Classification", "Semantic Search"])
_add("Word2Vec", NLP, ["Word2Vec", "GloVe", "FastText"], "learning word vector representations", ["Sentence Embeddings", "NLP"])
_add("Prompt Engineering", NLP, ["prompt engineering", "prompt design"], "designing prompts to get reliable output from language models", ["LLMs"])
_add("Text Preprocessing", NLP, ["text preprocessing", "tokenization", "tokenisation", "stemming", "lemmatization", "lemmatisation", "stop word removal", "text cleaning"], "tokenising and cleaning text before analysis", ["NLP", "NLTK", "spaCy", "Data Cleaning"])
_add("Semantic Search", NLP, ["semantic search", "semantic similarity", "vector search", "cosine similarity", "text similarity", "FAISS"], "retrieving or comparing text by meaning using vector similarity", ["Sentence Embeddings", "TF-IDF", "RAG"])

# ---------------------------------------------------------------- Databases
_add("SQL", DB, ["SQL", "PL/SQL", "T-SQL", "structured query language"], "writing SQL queries to retrieve and manipulate relational data", ["Database Design", "ETL", "Data Analysis"])
_add("MySQL", DB, ["MySQL", "MariaDB"], "working with the MySQL relational database", ["PostgreSQL", "SQL Server", "SQL"], implies=["SQL"])
_add("PostgreSQL", DB, ["PostgreSQL", "Postgres"], "working with the PostgreSQL relational database", ["MySQL", "SQL Server", "SQL"], implies=["SQL"])
_add("SQL Server", DB, ["SQL Server", "Microsoft SQL Server", "MSSQL", "MS SQL"], "working with Microsoft SQL Server", ["MySQL", "PostgreSQL", "SQL"], implies=["SQL"])
_add("Oracle Database", DB, ["Oracle Database", "Oracle DB"], "working with Oracle relational databases", ["SQL Server", "PostgreSQL", "SQL"], implies=["SQL"])
_add("SQLite", DB, ["SQLite"], "using the embedded SQLite database", ["MySQL", "SQL"], implies=["SQL"])
_add("MongoDB", DB, ["MongoDB", "Mongo DB"], "working with the MongoDB document database", ["NoSQL"], implies=["NoSQL"])
_add("NoSQL", DB, ["NoSQL", "non-relational database", "non-relational databases"], "using non-relational databases such as document or key-value stores", ["MongoDB", "Redis"])
_add("Redis", DB, ["Redis"], "using Redis as an in-memory key-value store or cache", ["NoSQL"], implies=["NoSQL"])
_add("Elasticsearch", DB, ["Elasticsearch", "Elastic Search", "OpenSearch"], "indexing and searching documents with Elasticsearch", ["Semantic Search", "NoSQL"])
_add("Snowflake", DB, ["Snowflake"], "using the Snowflake cloud data warehouse", ["Data Warehousing", "SQL"], implies=["Data Warehousing"])
_add("Data Warehousing", DB, ["data warehouse", "data warehouses", "data warehousing", "BigQuery", "Redshift"], "designing and querying analytical data warehouses", ["SQL", "ETL", "Snowflake"])
_add("ETL", DB, ["ETL", "ELT", "data pipelines", "data pipeline", "ETL pipelines"], "building extract, transform and load data pipelines", ["Apache Airflow", "SQL", "Data Cleaning", "Data Warehousing", "Apache Spark"])
_add("Database Design", DB, ["database design", "schema design", "data modeling", "data modelling", "ER diagram", "database normalization"], "designing relational schemas and data models", ["SQL"])

# ---------------------------------------------------------------- Cloud & DevOps
_add("Cloud Computing", CLOUD, ["cloud computing", "cloud platforms", "cloud services", "cloud infrastructure"], "building and running applications on cloud platforms", ["AWS", "Azure", "GCP"])
_add("AWS", CLOUD, ["AWS", "Amazon Web Services", "EC2", "SageMaker"], "using Amazon Web Services for compute, storage and deployment", ["Azure", "GCP", "Cloud Computing"], implies=["Cloud Computing"])
_add("Azure", CLOUD, ["Azure", "Microsoft Azure"], "using Microsoft Azure cloud services", ["AWS", "GCP", "Cloud Computing"], implies=["Cloud Computing"])
_add("GCP", CLOUD, ["GCP", "Google Cloud", "Google Cloud Platform"], "using Google Cloud Platform services", ["AWS", "Azure", "Cloud Computing"], implies=["Cloud Computing"])
_add("Docker", CLOUD, ["Docker", "containerization", "containerisation"], "packaging applications into Docker containers", ["Kubernetes", "CI/CD", "Linux"])
_add("Kubernetes", CLOUD, ["Kubernetes", "K8s"], "orchestrating containers with Kubernetes", ["Docker", "Cloud Computing"])
_add("Git", CLOUD, ["Git", "version control", "source control"], "tracking code changes with Git version control", ["GitHub"])
_add("GitHub", CLOUD, ["GitHub", "GitLab", "Bitbucket"], "collaborating on code with GitHub or GitLab", ["Git", "CI/CD"])
_add("Linux", CLOUD, ["Linux", "Ubuntu", "Unix"], "working on Linux or Unix command-line systems", ["Bash/Shell Scripting"])
_add("CI/CD", CLOUD, ["CI/CD", "CICD", "continuous integration", "continuous delivery", "continuous deployment", "Jenkins", "GitHub Actions", "GitLab CI"], "automating build, test and deployment pipelines", ["Docker", "GitHub", "Unit Testing"])
_add("Terraform", CLOUD, ["Terraform", "infrastructure as code", "!IaC"], "defining cloud infrastructure as code", ["AWS", "Azure", "GCP", "CI/CD"])
_add("Apache Airflow", CLOUD, ["Airflow", "Apache Airflow"], "scheduling and orchestrating data workflows with Airflow", ["ETL", "Python"])
_add("Apache Spark", CLOUD, ["Apache Spark", "PySpark", "!Spark"], "processing large datasets with Apache Spark", ["Hadoop", "ETL", "Scala"])
_add("Apache Kafka", CLOUD, ["Kafka", "Apache Kafka"], "streaming data with Apache Kafka", ["Apache Spark", "ETL"])
_add("Hadoop", CLOUD, ["Hadoop", "HDFS", "MapReduce"], "processing big data with the Hadoop ecosystem", ["Apache Spark"])

# ---------------------------------------------------------------- Web development
_add("HTML", WEB, ["HTML", "HTML5"], "structuring web pages with HTML", ["CSS", "JavaScript"])
_add("CSS", WEB, ["CSS", "CSS3", "SCSS", "Sass", "Tailwind"], "styling web pages with CSS", ["HTML"])
_add("JavaScript", WEB, ["JavaScript", "ECMAScript", "ES6"], "writing interactive web code in JavaScript", ["TypeScript", "HTML", "Node.js"])
_add("React", WEB, ["!React", "React.js", "ReactJS"], "building user interfaces with React", ["JavaScript", "Angular", "Vue.js"], implies=["JavaScript"])
_add("Angular", WEB, ["!Angular", "AngularJS"], "building web applications with Angular", ["JavaScript", "TypeScript", "React"], implies=["JavaScript"])
_add("Vue.js", WEB, ["Vue.js", "VueJS", "!Vue"], "building user interfaces with Vue.js", ["JavaScript", "React"], implies=["JavaScript"])
_add("Node.js", WEB, ["Node.js", "NodeJS"], "building server-side applications with Node.js", ["JavaScript", "Express.js"], implies=["JavaScript"])
_add("Express.js", WEB, ["Express.js", "ExpressJS"], "building web servers with Express.js", ["Node.js", "REST APIs"], implies=["JavaScript"])
_add("Flask", WEB, ["Flask"], "building web applications with Flask", ["Django", "FastAPI", "REST APIs"], implies=["Python"])
_add("Django", WEB, ["Django"], "building web applications with Django", ["Flask", "FastAPI", "REST APIs"], implies=["Python"])
_add("FastAPI", WEB, ["FastAPI"], "building APIs with FastAPI", ["Flask", "Django", "REST APIs"], implies=["Python"])
_add("REST APIs", WEB, ["REST API", "REST APIs", "RESTful", "RESTful API", "RESTful APIs", "API development", "API design"], "designing and consuming RESTful web APIs", ["Flask", "FastAPI", "Django", "Express.js", "Spring Boot", "GraphQL"])
_add("GraphQL", WEB, ["GraphQL"], "building and querying GraphQL APIs", ["REST APIs"])
_add("Streamlit", WEB, ["Streamlit"], "building interactive data apps with Streamlit", ["Flask", "Data Visualization"], implies=["Python"])
_add("Spring Boot", WEB, ["Spring Boot", "Spring Framework"], "building Java back-end services with Spring Boot", ["Java", "REST APIs"], implies=["Java"])
_add(".NET", WEB, [".NET", "ASP.NET", "dotnet"], "building applications on the .NET platform", ["C#"])

# ---------------------------------------------------------------- Software engineering
_add("Object-Oriented Programming", SE, ["object-oriented programming", "object oriented programming", "object-oriented design", "OOP", "OOPS"], "designing software with classes, inheritance and encapsulation", ["Java", "C++", "Design Patterns"])
_add("Data Structures & Algorithms", SE, ["data structures", "algorithms", "!DSA"], "choosing and implementing data structures and algorithms", ["Object-Oriented Programming"])
_add("System Design", SE, ["system design", "software architecture", "distributed systems", "scalable systems"], "designing scalable software systems", ["Microservices", "Design Patterns", "Cloud Computing"])
_add("Microservices", SE, ["microservices", "microservice architecture"], "building systems from small independently deployable services", ["REST APIs", "Docker", "System Design"])
_add("Design Patterns", SE, ["design patterns"], "applying established object-oriented design patterns", ["Object-Oriented Programming", "System Design"])
_add("Unit Testing", SE, ["unit testing", "unit tests", "pytest", "JUnit", "test-driven development", "!TDD", "automated testing", "test automation", "Selenium"], "writing automated tests to verify code", ["CI/CD", "Debugging"])
_add("Agile", SE, ["Agile", "Scrum", "Kanban", "sprint planning"], "working in iterative agile or scrum teams", ["Project Management", "Jira", "Teamwork"])
_add("Jira", SE, ["Jira", "Confluence"], "tracking work items with Jira", ["Agile"])
_add("Code Review", SE, ["code review", "code reviews", "pull requests", "pull request"], "reviewing and giving feedback on teammates' code", ["Git", "GitHub", "Teamwork"])
_add("Debugging", SE, ["debugging", "troubleshooting", "root cause analysis"], "finding and fixing defects in software", ["Unit Testing"])
_add("Software Development Lifecycle", SE, ["SDLC", "software development life cycle", "software development lifecycle"], "following the stages of the software development lifecycle", ["Agile"])

# ---------------------------------------------------------------- Visualization & analytics
_add("Data Visualization", VIZ, ["data visualization", "data visualisation", "visual analytics", "dashboards", "dashboard", "charts and graphs"], "presenting data with charts and dashboards",
     ["Streamlit", "Data Analysis", "Excel"])
_add("Tableau", VIZ, ["Tableau"], "building dashboards with Tableau", ["Power BI", "Looker", "Data Visualization"], implies=["Data Visualization"])
_add("Power BI", VIZ, ["Power BI", "PowerBI"], "building reports and dashboards with Power BI", ["Tableau", "Looker", "Data Visualization", "Excel"], implies=["Data Visualization"])
_add("Matplotlib", VIZ, ["Matplotlib"], "plotting charts with Matplotlib", ["Seaborn", "Plotly"], implies=["Data Visualization", "Python"])
_add("Seaborn", VIZ, ["Seaborn"], "creating statistical plots with Seaborn", ["Matplotlib"], implies=["Data Visualization", "Python"])
_add("Plotly", VIZ, ["Plotly"], "creating interactive charts with Plotly", ["Matplotlib", "Streamlit"], implies=["Data Visualization"])
_add("Looker", VIZ, ["Looker", "Looker Studio", "Data Studio"], "building dashboards with Looker", ["Tableau", "Power BI", "Data Visualization"], implies=["Data Visualization"])
_add("Excel", VIZ, ["!Excel", "Microsoft Excel", "spreadsheets", "pivot tables", "VLOOKUP"], "analysing data in spreadsheets with Excel", ["Data Analysis", "Power BI"])
_add("Business Intelligence", VIZ, ["business intelligence", "!BI", "BI tools", "KPI reporting"], "turning business data into reports and KPIs", ["Tableau", "Power BI", "Data Visualization", "SQL"])
_add("Data Storytelling", VIZ, ["data storytelling", "storytelling with data"], "communicating insights from data to an audience", ["Data Visualization", "Presentation Skills"])

# ---------------------------------------------------------------- Communication & teamwork
_add("Communication", SOFT, ["communication", "communication skills", "communicate effectively", "verbal communication", "written communication"], "communicating clearly in speech and writing", ["Presentation Skills", "Stakeholder Management", "Teamwork"])
_add("Teamwork", SOFT, ["teamwork", "team player", "collaboration", "collaborated", "collaborate", "cross-functional teams", "cross-functional", "worked in teams", "work in a team", "work in teams", "work well in teams", "work collaboratively"], "collaborating with others in a team", ["Communication", "Leadership", "Agile"])
_add("Leadership", SOFT, ["leadership", "led a team", "led teams", "team lead", "team leader", "team leadership", "leading a team", "managed a team"], "leading and motivating a team", ["Project Management", "Mentoring", "Teamwork"])
_add("Problem Solving", SOFT, ["problem solving", "problem-solving", "analytical skills", "analytical thinking", "critical thinking"], "analysing problems and developing solutions", ["Debugging", "Data Analysis"])
_add("Project Management", SOFT, ["project management", "managing projects", "project planning", "!PMP"], "planning and delivering projects on time", ["Agile", "Leadership", "Jira"])
_add("Time Management", SOFT, ["time management", "meeting deadlines", "prioritization", "prioritisation"], "organising work and meeting deadlines", ["Project Management"])
_add("Presentation Skills", SOFT, ["presentation skills", "presentations", "public speaking", "presenting", "presented"], "presenting ideas and results to an audience", ["Communication", "Data Storytelling"])
_add("Mentoring", SOFT, ["mentoring", "mentorship", "mentored", "coaching"], "guiding and teaching less experienced people", ["Leadership", "Teamwork"])
_add("Stakeholder Management", SOFT, ["stakeholder management", "stakeholder communication", "stakeholder engagement", "stakeholders"], "working with stakeholders to understand and manage expectations", ["Communication", "Project Management"])


def validate_dictionary() -> list[str]:
    """Return a list of integrity problems (empty list means the dictionary is consistent)."""
    problems: list[str] = []
    seen_alias: dict[str, str] = {}
    for name, e in SKILLS.items():
        if e.category not in CATEGORIES:
            problems.append(f"{name}: unknown category {e.category}")
        if not e.aliases:
            problems.append(f"{name}: no aliases")
        for rel in (*e.related, *e.implies):
            if rel not in SKILLS:
                problems.append(f"{name}: relationship to unknown skill {rel}")
            if rel == name:
                problems.append(f"{name}: relates to itself")
        for a in e.aliases:
            key = a.lstrip("!").lower()
            if key in seen_alias and seen_alias[key] != name:
                problems.append(f"alias '{a}' used by both {seen_alias[key]} and {name}")
            seen_alias[key] = name
    return problems


def skills_by_category() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    for e in SKILLS.values():
        out[e.category].append(e.name)
    return out
