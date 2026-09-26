import json
import re
from typing import List

from app.core.config import settings


ABBREVIATIONS = {
    "py": "Python",
    "python": "Python",
    "js": "JavaScript",
    "javascript": "JavaScript",
    "ts": "TypeScript",
    "ml": "Machine Learning",
    "ai": "Artificial Intelligence",
    "cv": "Computer Vision",
    "nlp": "Natural Language Processing",
    "sql": "SQL database",
    "cgpa": "grades academic performance",
    "srm": "SRM Institute of Science and Technology",
}

TYPO_CORRECTIONS = {
    "hooby": "hobby",
    "hobies": "hobbies",
    "sumamry": "summary",
    "experiance": "experience",
    "skils": "skills",
    "educaton": "education",
}

STOPWORDS = {
    "a", "an", "and", "are", "does", "he", "how", "i", "is", "it", "know",
    "me", "of", "she", "tell", "the", "this", "to", "what", "with", "you",
}


def preprocess_query(query: str) -> List[str]:
    """Return original, keyword-focused, and semantic query variants."""
    normalized = _normalize_query(query)
    fallback = _deterministic_variations(normalized)

    provider = settings.LLM_PROVIDER.lower()
    if not ((provider == "gemini" and settings.GEMINI_API_KEY) or
            (provider == "openai" and settings.OPENAI_API_KEY)):
        return fallback

    rewritten = _call_rewriter(normalized, fallback)
    return _validate_variations(normalized, rewritten or fallback)


def _normalize_query(query: str) -> str:
    normalized = re.sub(r"\s+", " ", query.strip())
    for typo, correction in TYPO_CORRECTIONS.items():
        normalized = re.sub(rf"\b{re.escape(typo)}\b", correction, normalized, flags=re.IGNORECASE)
    return normalized


def _deterministic_variations(query: str) -> List[str]:
    tokens = re.findall(r"[\w-]+", query.lower())
    expanded_tokens = [ABBREVIATIONS.get(token, token) for token in tokens]
    keyword_terms = [token for token in expanded_tokens if token.lower() not in STOPWORDS]
    keyword_query = " ".join(keyword_terms) or query

    semantic_query = query
    lowered = query.lower()
    if re.search(r"\bdoes\s+(he|she|the candidate)\s+know\b", lowered):
        subject = re.search(r"\b(he|she|the candidate)\b", lowered).group(1)
        term = next((ABBREVIATIONS[token] for token in tokens if token in ABBREVIATIONS), keyword_query)
        semantic_query = f"Does the {subject if subject == 'candidate' else 'candidate'} have skills or experience in {term}?"
    elif "cgpa" in lowered:
        semantic_query = "What are the candidate's CGPA, grades, marks, or academic performance?"
    elif "hobby" in lowered or "hobbies" in lowered:
        semantic_query = "What hobbies and interests are listed in the document?"
    elif re.search(r"\bname\b", lowered):
        semantic_query = "What is the person's full name or candidate name?"
    elif re.search(r"\bemail\b", lowered):
        semantic_query = "What email address or contact email is listed?"
    elif re.search(r"\bphone\b|\bmobile\b|\bnumber\b", lowered):
        semantic_query = "What phone number or mobile contact number is listed?"
    elif re.search(r"\bdetails?\b", lowered):
        semantic_query = "What are the person's profile, contact, education, skills, and experience details?"

    variations = _unique([query, keyword_query, semantic_query])
    while len(variations) < 3:
        variations.append(f"{keyword_query} information details {len(variations)}")
    return _unique(variations)[:3]


def _call_rewriter(query: str, fallback: List[str]) -> List[str] | None:
    prompt = (
        "You rewrite document-search queries. Return JSON only in this shape: "
        '{"original":"...","keyword":"...","semantic":"..."}. '
        "Correct obvious typos, expand abbreviations such as py=Python, js=JavaScript, "
        "ML=Machine Learning, and CGPA=grades/marks/academic performance. "
        "Keep the original intent, make the keyword version lexical, and make the semantic "
        "version a clear question. Do not add facts or answer the question. Query: " + query
    )
    try:
        if settings.LLM_PROVIDER.lower() == "gemini":
            from google import genai
            response = genai.Client(api_key=settings.GEMINI_API_KEY).models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
            )
            raw = response.text
        else:
            from openai import OpenAI
            response = OpenAI(api_key=settings.OPENAI_API_KEY).chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content

        payload = json.loads(_extract_json(raw))
        return _unique([payload.get("original", ""), payload.get("keyword", ""), payload.get("semantic", "")])
    except Exception:
        return fallback


def _extract_json(raw: str) -> str:
    match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
    if not match:
        raise ValueError("Query rewriter did not return JSON")
    return match.group(0)


def _validate_variations(original: str, variations: List[str]) -> List[str]:
    safe = [original]
    safe.extend(item.strip() for item in variations if isinstance(item, str) and item.strip())
    validated = _unique(safe)
    while len(validated) < 3:
        validated.append(f"{original} information details {len(validated)}")
    return _unique(validated)[:3]


def _unique(values: List[str]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        cleaned = re.sub(r"\s+", " ", value.strip())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result
