import spacy
import re

nlp = spacy.load("en_core_web_sm")

EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_RE = re.compile(r'(\+?1?\s?)?(\(?\d{3}\)?[\s.\-]?)(\d{3}[\s.\-]?\d{4})')
URL_RE = re.compile(r'https?://\S+|linkedin\.com/\S+|github\.com/\S+')

SKILLS = [
    "Python", "Java", "SQL", "Go", "TypeScript", "Docker", "Kubernetes",
    "AWS", "GCP", "Azure", "PyTorch", "TensorFlow", "scikit-learn",
    "FastAPI", "Flask", "React", "Kafka", "Spark", "Airflow", "MLflow",
    "LangChain", "GPT", "BERT", "RAG", "LLM", "NLP", "Computer Vision",
    "Machine Learning", "Deep Learning", "XGBoost", "Snowflake", "dbt",
    "Pandas", "NumPy", "SHAP", "HuggingFace", "LoRA", "ONNX",
    "JavaScript", "C++", "C#", "Ruby", "Rust", "Scala", "R",
    "Node.js", "Django", "Spring", "PostgreSQL", "MongoDB", "Redis",
    "Terraform", "Jenkins", "Git", "CI/CD", "REST API", "GraphQL",
    "Tableau", "Power BI", "Excel", "Jira", "Agile", "Scrum",
]


def parse_resume(text: str) -> dict:
    """Parse resume text into structured fields using spaCy NER and regex."""
    doc = nlp(text)

    persons = [e.text for e in doc.ents if e.label_ == "PERSON"]
    orgs = [e.text for e in doc.ents if e.label_ == "ORG"]
    locations = [e.text for e in doc.ents if e.label_ in ("GPE", "LOC")]
    dates = [e.text for e in doc.ents if e.label_ == "DATE"]

    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    urls = URL_RE.findall(text)

    # Case-insensitive skill matching
    text_lower = text.lower()
    skills = [s for s in SKILLS if s.lower() in text_lower]

    return {
        "name": persons[0] if persons else "Not detected",
        "email": emails[0] if emails else "Not detected",
        "phone": "".join(phones[0]) if phones else "Not detected",
        "location": locations[0] if locations else "Not detected",
        "links": urls[:5],
        "companies": list(dict.fromkeys(orgs))[:8],  # preserve order, dedupe
        "skills": skills,
        "dates": list(set(dates))[:10],
        "all_persons": persons,
        "all_locations": locations,
    }