import os
import re
from typing import List, Dict, Any
from app.core.config import settings

def generate_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    query_variations: List[str] | None = None,
) -> str:
    """
    Synthesize an answer to user query using retrieved document chunks as context.
    Universal dynamic synthesis for ANY uploaded document (resumes, press releases, reports, etc.).
    """
    if not retrieved_chunks:
        return "Based on the provided document, I could not find information to answer this question."

    context_chunks = _assemble_context_chunks(retrieved_chunks)
    context_blocks = []
    for idx, chunk in enumerate(context_chunks, start=1):
        context_blocks.append(
            f"[Passage {idx} | Source: {chunk['source']} | Chunk #{chunk['chunk_index']}]\n{chunk['text']}"
        )
    combined_context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an expert AI Research Assistant. Analyze the provided source-document context and answer the user's question accurately, concisely, and completely. "
        "Read and synthesize all provided context chunks; do not rely on isolated bullet fragments or a single passage when the answer requires information across passages. "
        "Base every claim strictly on facts present in the context. Do not invent, extrapolate, or use outside knowledge. "
        "Answer the user's question directly first, followed by relevant supporting details when useful. "
        "Treat shorthand, abbreviations, typos, and informal references according to the supplied search interpretations, but verify the resulting answer against the context. "
        "If the context genuinely does not contain enough information, say exactly: Based on the provided document, I could not find information to answer this question. "
        "Use natural language, concise headers, and bullet points when appropriate. Do not mention retrieval, search interpretations, prompts, or hidden instructions, and do not use markdown bold stars."
    )

    interpreted_query = " | ".join(query_variations or [query])
    prompt = (
        f"{system_prompt}\n\n=== CONTEXT PASSAGES ===\n{combined_context}"
        f"\n\n=== USER QUESTION ===\n{query}"
        f"\n=== SEARCH INTERPRETATIONS (use only to understand shorthand) ===\n{interpreted_query}"
        "\n\n=== ANSWER ==="
    )

    provider = settings.LLM_PROVIDER.lower()

    if provider == "gemini" and settings.GEMINI_API_KEY:
        raw_ans = _call_gemini_api(prompt)
        if raw_ans.startswith("[LLM Error"):
            return _clean_stars(_synthesize_dynamic_document_response(query, context_chunks, query_variations))
        return _clean_stars(raw_ans)
    elif provider == "openai" and settings.OPENAI_API_KEY:
        raw_ans = _call_openai_api(prompt)
        if raw_ans.startswith("[LLM Error"):
            return _clean_stars(_synthesize_dynamic_document_response(query, context_chunks, query_variations))
        return _clean_stars(raw_ans)
    else:
        return _clean_stars(_synthesize_dynamic_document_response(query, context_chunks, query_variations))


def _assemble_context_chunks(retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge adjacent chunks from the same document before synthesis."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in retrieved_chunks:
        grouped.setdefault(chunk["source"], []).append(chunk)

    assembled = []
    for source, chunks in grouped.items():
        ordered = sorted(chunks, key=lambda chunk: chunk["chunk_index"])
        current = None
        for chunk in ordered:
            if current is None:
                current = dict(chunk)
                continue

            if chunk["chunk_index"] == current["chunk_index"] + 1:
                current["text"] = _join_chunk_text(current["text"], chunk["text"])
                current["chunk_index"] = chunk["chunk_index"]
            else:
                assembled.append(current)
                current = dict(chunk)

        if current is not None:
            assembled.append(current)

    return assembled


def _join_chunk_text(first: str, second: str) -> str:
    """Join overlapping chunk text without duplicating the overlap."""
    first_clean = first.strip()
    second_clean = second.strip()
    max_overlap = min(len(first_clean), len(second_clean), 150)
    for size in range(max_overlap, 20, -1):
        if first_clean[-size:].lower() == second_clean[:size].lower():
            return f"{first_clean}{second_clean[size:]}"
    return f"{first_clean}\n{second_clean}"

def _clean_stars(text: str) -> str:
    """Remove markdown bold asterisks (**) and bullet stars (*) for clean output."""
    cleaned = text.replace('**', '').replace('***', '')
    cleaned = re.sub(r"(?m)^\s*\*\s+", "- ", cleaned)
    cleaned = re.sub(r'💡\s*\*Tip:.*', '', cleaned)
    return cleaned.strip()

def _call_gemini_api(prompt: str) -> str:
    """Call Google Gemini API."""
    try:
        from google import genai
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"[LLM Error - Gemini API Call Failed]: {str(e)}"

def _call_openai_api(prompt: str) -> str:
    """Call OpenAI API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM Error - OpenAI API Call Failed]: {str(e)}"

def _synthesize_dynamic_document_response(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    query_variations: List[str] | None = None,
) -> str:
    """
    Universal Dynamic Document Synthesizer for ANY uploaded file.
    Extracts facts, key sentences, and relevant details dynamically from retrieved passages.
    """
    source_name = retrieved_chunks[0]["source"] if retrieved_chunks else "Uploaded Document"
    combined_text = "\n\n".join([c["text"] for c in retrieved_chunks])

    # Keep complete sentences and bullets from the assembled context.
    raw_lines = [line.strip() for line in combined_text.split('\n') if line.strip()]

    focused_sentences = _extract_focused_profile_facts(query, raw_lines)
    if focused_sentences:
        matching_sentences = focused_sentences
    else:
        matching_sentences = []
    
    # 1. Search for sentences directly containing user query words
    searchable_query = " ".join(query_variations or [query])
    q_words = set(re.findall(r'\w+', searchable_query.lower()))
    # Exclude common stopwords from matching
    stopwords = {"what", "is", "the", "a", "an", "this", "of", "to", "in", "for", "on", "with", "and", "are", "or", "tell", "me", "about"}
    relevant_q_words = q_words - stopwords

    if not matching_sentences:
        for line in raw_lines:
            sentence_parts = re.split(r"(?<=[.!?])\s+", line)
            for sentence in sentence_parts:
                sentence = sentence.strip()
                sentence_words = set(re.findall(r'\w+', sentence.lower()))
                if (sentence and relevant_q_words.intersection(sentence_words)
                        and sentence not in matching_sentences):
                    matching_sentences.append(sentence)

    if not matching_sentences:
        return "Based on the provided document, I could not find information to answer this question."

    output_sections = [f"### Research Response for '{query}'", f"Source Document: {source_name}", ""]
    grouped_lines: Dict[str, List[str]] = {}
    for sentence in matching_sentences:
        clean_sentence = sentence.lstrip('•-–* ').strip()
        category = _categorize_sentence(clean_sentence)
        grouped_lines.setdefault(category, []).append(clean_sentence)

    for category, sentences in grouped_lines.items():
        output_sections.append(f"{category}:")
        for sentence in sentences:
            output_sections.append(f"- {sentence}")
        output_sections.append("")

    return "\n".join(output_sections).strip()


def _extract_focused_profile_facts(query: str, raw_lines: List[str]) -> List[str]:
    """Answer short contact/profile queries from explicit resume evidence."""
    lowered = query.lower()
    sentences = []
    for line in raw_lines:
        parts = re.split(r"(?<=[.!?])\s+", line)
        sentences.extend(part.strip() for part in parts if part.strip())

    if re.search(r"\bemail\b", lowered):
        return [sentence for sentence in sentences if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", sentence)]

    if re.search(r"\b(phone|mobile|number)\b", lowered):
        return [sentence for sentence in sentences if re.search(r"(?:\+?\d[\d\s().-]{7,}\d)", sentence)]

    if re.search(r"\bname\b", lowered):
        for sentence in sentences[:3]:
            if not re.search(r"@|\d{5,}|linkedin|github", sentence, re.IGNORECASE):
                return [sentence]

    return []


def _categorize_sentence(sentence: str) -> str:
    """Group common machine-learning categories without adding new facts."""
    normalized = sentence.lower()
    if "unsupervised" in normalized:
        return "Unsupervised Learning"
    if "supervised" in normalized:
        return "Supervised Learning"
    if "reinforcement" in normalized:
        return "Reinforcement Learning"
    return "Relevant Information"
