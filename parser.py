import spacy
import re

nlp = spacy.load("en_core_web_sm")

EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_RE = re.compile(r'(\+?1?\s?)?(\(?\d{3}\)?[\s.\-]?)(\d{3}[\s.\-]?\d{4})')
URL_RE = re.compile(r'https?://\S+|linkedin\.com/\S+|github\.com/\S+')

# Common technical terms that spaCy misclassifies as PERSON/ORG
FALSE_PERSON_TERMS = {
    "stacked ensembles", "kernelexplainer", "treeexplainer", "pyspark",
    "pandas", "kafka", "snowflake", "docker", "streamlit", "pytorch",
    "tensorflow", "scikit-learn", "xgboost", "fastapi", "langchain",
    "langgraph", "llama", "bart", "flan-t5", "numpy", "shap",
    "airflow", "mlflow", "kubernetes", "rabbitmq", "minio",
    "tableau", "spark", "rag", "lora", "huggingface", "onnx",
    "bert", "gpt", "gpt-4", "clip", "lancedb", "svd", "rouge",
    "bleu", "flask", "react", "redis", "mongodb", "postgresql",
    "terraform", "jenkins", "c++", "scala", "python", "java", "sql",
    "developed tableau",
}

FALSE_ORG_TERMS = {
    "machine learning engineer", "data scientist", "application engineer",
    "ml", "ai", "etl", "ci/cd", "qa", "gpa", "cgpa",
    "shap", "rag", "llm", "nlp", "svd", "rmse",
    "h2o automl", "automl",
}

FALSE_LOCATION_TERMS = {
    "numpy", "lambda", "qa", "s3", "ec2", "sagemaker",
}

# US state abbreviations for location detection
US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

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
    "PySpark", "H2O AutoML", "LangGraph", "RabbitMQ",
    "SageMaker", "Lambda", "S3", "EC2",
    "Transformers", "BART", "Flan-T5", "CLIP", "LanceDB",
]


def _is_valid_person(name: str) -> bool:
    """Filter out false positive person detections."""
    name_lower = name.lower().strip()
    if name_lower in FALSE_PERSON_TERMS:
        return False
    if "\n" in name:
        return False
    # Reject single words that look like tech terms
    if len(name.split()) == 1:
        if name[0].islower() or name.isupper():
            return False
    # Reject if contains special characters common in tech but not names
    if any(c in name for c in "()@/\\{}[]<>=+*&^%$#!~`"):
        return False
    return True


def _is_valid_org(org: str) -> bool:
    """Filter out false positive organization detections."""
    org_lower = org.lower().strip()
    if org_lower in FALSE_ORG_TERMS:
        return False
    if "\n" in org:
        return False
    if len(org) > 60:
        return False
    # Reject concatenated strings (no spaces, long)
    if len(org) > 15 and " " not in org:
        return False
    if "@" in org or "⋄" in org:
        return False
    return True


def _is_valid_location(loc: str) -> bool:
    """Filter out false positive location detections."""
    loc_lower = loc.lower().strip()
    if loc_lower in FALSE_LOCATION_TERMS:
        return False
    if "\n" in loc:
        return False
    return True


def _extract_name_from_top(text: str) -> str | None:
    """
    Extract candidate name from the first few lines of the resume.
    Most resumes have the name as the first non-empty line.
    """
    lines = text.strip().split("\n")
    for line in lines[:5]:
        line = line.strip()
        if not line:
            continue
        # Skip contact info lines
        if "@" in line or "linkedin" in line.lower() or "github" in line.lower():
            continue
        if PHONE_RE.search(line):
            continue
        # A name line is typically short and mostly alphabetic
        if len(line) < 50:
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in line) / max(len(line), 1)
            if alpha_ratio > 0.8 and len(line.split()) >= 2:
                return line.strip()
    return None


def _expand_location(loc: str, text: str) -> str:
    """Try to expand a state abbreviation to 'City, State' from context."""
    if loc.upper() in US_STATES and len(loc) <= 3:
        pattern = re.compile(rf'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*),?\s*{re.escape(loc)}')
        match = pattern.search(text)
        if match:
            return f"{match.group(1)}, {loc}"
    return loc


def parse_resume(text: str) -> dict:
    """Parse resume text into structured fields using spaCy NER and regex."""
    doc = nlp(text)

    # Extract and filter entities
    raw_persons = [e.text for e in doc.ents if e.label_ == "PERSON"]
    raw_orgs = [e.text for e in doc.ents if e.label_ == "ORG"]
    raw_locations = [e.text for e in doc.ents if e.label_ in ("GPE", "LOC")]
    dates = [e.text.strip() for e in doc.ents if e.label_ == "DATE"]

    persons = [p for p in raw_persons if _is_valid_person(p)]
    orgs = [o for o in raw_orgs if _is_valid_org(o)]
    locations = [loc for loc in raw_locations if _is_valid_location(loc)]

    # Get name from top of resume first (most reliable for resumes)
    top_name = _extract_name_from_top(text)

    # Regex extractions
    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    urls = URL_RE.findall(text)

    # Case-insensitive skill matching
    text_lower = text.lower()
    skills = [s for s in SKILLS if s.lower() in text_lower]

    # Determine best name
    name = top_name or (persons[0] if persons else "Not detected")

    # Expand location if it's just a state abbreviation
    location = locations[0] if locations else "Not detected"
    if location != "Not detected":
        location = _expand_location(location, text)

    # Clean dates
    clean_dates = [d for d in dates if "\n" not in d and len(d) < 40]

    return {
        "name": name,
        "email": emails[0] if emails else "Not detected",
        "phone": "".join(phones[0]) if phones else "Not detected",
        "location": location,
        "links": urls[:5],
        "companies": list(dict.fromkeys(orgs))[:8],
        "skills": skills,
        "dates": list(set(clean_dates))[:10],
        "all_persons": persons,
        "all_locations": locations,
    }