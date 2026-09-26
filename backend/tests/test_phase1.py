import os
import sys
from pathlib import Path

# Add backend directory to sys.path so imports work cleanly
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from main import app
from app.services.extractor import extract_text_from_file
from app.services.chunker import chunk_text
from app.services.vector_store import vector_store

client = TestClient(app)

def run_phase1_tests():
    print("=" * 60)
    print("RUNNING PHASE 1 VERIFICATION TESTS: BASIC RAG PIPELINE")
    print("=" * 60)

    # 0. Clear previous vector store data for isolated testing
    vector_store.clear_all()
    print("[1/5] Vector store cleared successfully.")

    # 1. Test Health Endpoint
    response = client.get("/health")
    assert response.status_code == 200, f"Health check failed: {response.text}"
    data = response.json()
    print(f"[2/5] Health Check PASSED: status='{data['status']}', indexed_chunks={data['document_chunks_indexed']}")

    # 2. Create a temporary sample document (.txt) for testing ingestion
    sample_filename = "sample_ai_paper.txt"
    sample_file_path = os.path.join(backend_dir, "storage", "uploads", sample_filename)
    
    sample_content = (
        "Retrieval-Augmented Generation (RAG) is an AI architecture that combines "
        "information retrieval systems with generative large language models (LLMs). "
        "Instead of relying purely on static parametric knowledge memorized during training, "
        "RAG dynamically fetches relevant external documents from a vector database or search index.\n\n"
        "Vector embeddings transform unstructured textual data into high-dimensional numerical vectors. "
        "Similar concepts are positioned close to each other in vector space using distance metrics like "
        "cosine similarity or Euclidean distance.\n\n"
        "ChromaDB is an open-source vector store designed for AI applications. It enables storing "
        "embeddings along with document metadata and performing sub-second similarity lookups."
    )
    
    with open(sample_file_path, "w", encoding="utf-8") as f:
        f.write(sample_content)

    # 3. Test Text Extraction & Chunking directly
    extracted_text = extract_text_from_file(sample_file_path, sample_filename)
    assert len(extracted_text) > 0, "Text extraction returned empty string!"
    chunks = chunk_text(extracted_text, source_name=sample_filename, chunk_size=200, chunk_overlap=30)
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
    print(f"[3/5] Extraction & Chunking PASSED: Extracted {len(extracted_text)} chars into {len(chunks)} chunks.")

    # 4. Test Upload Endpoint (/documents/upload)
    with open(sample_file_path, "rb") as f:
        upload_resp = client.post("/documents/upload", files={"file": (sample_filename, f, "text/plain")})
    
    assert upload_resp.status_code == 201, f"Document upload failed: {upload_resp.text}"
    upload_data = upload_resp.json()
    print(f"[4/5] API Upload PASSED: Indexed {upload_data['chunks_created']} chunks into vector store.")

    # 5. Test Chat Endpoint (/chat)
    query = "What is RAG and how does it work with vector databases?"
    chat_resp = client.post("/chat", json={"question": query, "top_k": 2})
    assert chat_resp.status_code == 200, f"Chat failed: {chat_resp.text}"
    chat_data = chat_resp.json()
    
    assert len(chat_data["sources"]) > 0, "Chat returned 0 source citations!"
    print(f"[5/5] API Chat PASSED!")
    print(f"      Question: '{chat_data['question']}'")
    print(f"      Top Citation: '{chat_data['sources'][0]['source']}' (Score: {chat_data['sources'][0]['similarity_score']})")
    print(f"      Answer Snippet: '{chat_data['answer'][:150]}...'")

    print("\n" + "=" * 60)
    print("ALL PHASE 1 TESTS PASSED SUCCESSFULLY! PROCEEDING TO EXPLANATION.")
    print("=" * 60)

if __name__ == "__main__":
    run_phase1_tests()
