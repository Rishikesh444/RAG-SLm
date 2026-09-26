from typing import List, Dict, Any

class RerankerManager:
    """
    Cross-Encoder Reranker using FlashRank ONNX model (ms-marco-TinyBERT-L-2-v2).
    Re-scores candidate passages returned by hybrid retrieval.
    Lazy-loads on first use to prevent startup crashes if ONNX DLLs are blocked.
    """
    def __init__(self):
        self._ranker = None
        self._available = None  # None = untested, True/False = known state

    def _get_ranker(self):
        """Lazy-initialize the FlashRank ranker on first use."""
        if self._available is False:
            return None
        if self._ranker is None:
            try:
                from flashrank import Ranker, RerankRequest  # noqa: F401
                self._ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
                self._available = True
                print("[Reranker] FlashRank loaded successfully.")
            except Exception as e:
                self._available = False
                print(f"[Reranker] WARNING: Could not load FlashRank ({e}). Reranking disabled.")
        return self._ranker

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Re-score and re-order candidate passages using cross-encoder attention.

        Args:
            query: User's question string.
            candidate_chunks: Candidates retrieved from hybrid vector + BM25 search.
            top_k: Number of final reranked passages to return.

        Returns:
            Sorted list of chunks with attached `rerank_score`.
            Falls back to returning top_k candidates as-is if reranker is unavailable.
        """
        if not candidate_chunks:
            return []

        ranker = self._get_ranker()

        # Graceful fallback: return top_k candidates without reranking
        if ranker is None:
            return candidate_chunks[:top_k]

        try:
            from flashrank import RerankRequest
            # Convert chunks into FlashRank passage structure
            passages = [
                {
                    "id": idx,
                    "text": chunk["text"],
                    "meta": chunk  # Attach original chunk dict as metadata
                }
                for idx, chunk in enumerate(candidate_chunks)
            ]

            rerank_request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(rerank_request)

            reranked_chunks = []
            for res in results[:top_k]:
                chunk_data = dict(res["meta"])
                # Convert numpy float to standard python float
                chunk_data["rerank_score"] = round(float(res["score"]), 5)
                reranked_chunks.append(chunk_data)

            return reranked_chunks

        except Exception as e:
            print(f"[Reranker] ERROR during reranking: {e}. Falling back to top_k candidates.")
            return candidate_chunks[:top_k]

# Global singleton instance
reranker = RerankerManager()
