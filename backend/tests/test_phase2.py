import os
import sys
from pathlib import Path

# Add backend directory to sys.path so imports work cleanly
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from main import app
from app.services.vector_store import vector_store
from app.services.bm25_retriever import bm25_retriever

client = TestClient(app)

def run_phase2_tests():
    print("=" * 60)
    print("RUNNING PHASE 2 VERIFICATION TESTS: HYBRID RETRIEVAL (VECTOR + BM25 + RRF)")
    print("=" * 60)

    # 0. Clear indices
    vector_store.clear_all()
    bm25_retriever.clear_all()
    print("[1/4] Indices cleared successfully.")

    # 1. Create a synthetic test file containing technical terms & general explanations
    sample_filename = "server_manual.txt"
    sample_path = os.path.join(backend_dir, "storage", "uploads", sample_filename)
    
    sample_content = (
        "Server Maintenance Guide 2026.\n\n"
        "Section A: Error Codes\n"
        "If the system raises error code ERR_8820_NETWORK_TIMEOUT, check the ethernet hardware interface.\n"
        "If the system raises error code ERR_9901_DATABASE_FAILURE, restart the PostgreSQL service container.\n\n"
        "Section B: Performance Tuning\n"
        "To optimize application response speeds, implement memory caching using Redis and database connection pooling.\n"
        "Decreasing query response latency is essential for maintaining optimal throughput under high traffic loads."
    )
    
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(sample_content)

    # 2. Upload sample document via API
    with open(sample_path, "rb") as f:
        upload_resp = client.post("/documents/upload", files={"file": (sample_filename, f, "text/plain")})
    
    assert upload_resp.status_code == 201, f"Upload failed: {upload_resp.text}"
    print(f"[2/4] Upload PASSED: Document ingested into ChromaDB and BM25 index.")

    # 3. Test Keyword Search (BM25 vs Hybrid on exact term "ERR_9901_DATABASE_FAILURE")
    keyword_query = "How to fix ERR_9901_DATABASE_FAILURE?"
    
    bm25_resp = client.post("/chat", json={"question": keyword_query, "top_k": 2, "retrieval_mode": "bm25"})
    assert bm25_resp.status_code == 200
    bm25_data = bm25_resp.json()
    assert "ERR_9901_DATABASE_FAILURE" in bm25_data["sources"][0]["text"]
    print(f"[3/4] BM25 Search PASSED: Matched exact code 'ERR_9901_DATABASE_FAILURE' accurately.")

    # 4. Test Hybrid Search (RRF Rank Fusion)
    hybrid_resp = client.post("/chat", json={"question": keyword_query, "top_k": 2, "retrieval_mode": "hybrid"})
    assert hybrid_resp.status_code == 200
    hybrid_data = hybrid_resp.json()
    
    assert hybrid_data["retrieval_mode"] == "hybrid"
    top_source = hybrid_data["sources"][0]
    assert top_source["rrf_score"] is not None
    print(f"[4/4] Hybrid Search PASSED!")
    print(f"      Retrieval Mode: '{hybrid_data['retrieval_mode']}'")
    print(f"      Top Match RRF Score: {top_source['rrf_score']}")
    print(f"      Matched Content Snippet: '{top_source['text'][:120]}...'")

    print("\n" + "=" * 60)
    print("ALL PHASE 2 TESTS PASSED SUCCESSFULLY! PROCEEDING TO EXPLANATION.")
    print("=" * 60)

if __name__ == "__main__":
    run_phase2_tests()
