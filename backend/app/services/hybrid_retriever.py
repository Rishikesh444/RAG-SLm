import hashlib
from typing import List, Dict, Any, Literal
from app.services.vector_store import vector_store
from app.services.bm25_retriever import bm25_retriever
from app.services.reranker import reranker
from app.core.config import settings

class HybridRetriever:
    """
    Hybrid Retriever combining Vector Search (ChromaDB) and Lexical Search (BM25)
    using Reciprocal Rank Fusion (RRF) and optional Cross-Encoder Reranking.
    Includes smart fallback to dense vector search if sparse BM25 keyword matching returns no results.
    """

    def _hash_text(self, text: str) -> str:
        """Helper to create MD5 content hash for deduplication."""
        return hashlib.md5(text.strip().encode('utf-8')).hexdigest()

    def retrieve(
        self, 
        query: str, 
        top_k: int = 4, 
        mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
        use_reranker: bool = True,
        rrf_k: int = 60,
        query_variations: List[str] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve context passages using specified mode with automatic fallback mechanism.
        """
        self._restore_bm25_index()
        candidate_k = 20 if use_reranker else top_k
        queries = query_variations or [query]
        candidates = self._retrieve_variations(queries, candidate_k, mode, rrf_k)

        # If query is a general query (like "what name", "summary", "what you found"),
        # and search results are still empty, fetch the top document chunks of the indexed document!
        if not candidates and vector_store.get_total_chunks() > 0:
            candidates = vector_store.similarity_search("overview document profile summary", top_k=candidate_k)

        # Deduplicate candidates by text content before reranking
        unique_candidates = []
        seen_hashes = set()
        for cand in candidates:
            h = self._hash_text(cand["text"])
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_candidates.append(cand)

        # Pass candidate pool through Cross-Encoder Reranker if enabled
        if use_reranker and unique_candidates:
            reranked_candidates = reranker.rerank(
                # Keep reranking anchored to the user's exact wording. The
                # expanded variants broaden recall, but broad semantic text
                # can demote exact contact terms such as email or phone.
                query=query,
                candidate_chunks=unique_candidates,
                top_k=top_k
            )
            threshold = settings.RERANK_RELEVANCE_THRESHOLD
            if threshold > 0 and any("rerank_score" in chunk for chunk in reranked_candidates):
                filtered_candidates = [
                    chunk for chunk in reranked_candidates
                    if chunk.get("rerank_score", 0.0) >= threshold
                ]
                # Generic short queries can receive uniformly low cross-encoder
                # scores even when lexical retrieval found valid evidence.
                # Keep the retrieval-ranked candidates rather than returning no context.
                if filtered_candidates:
                    reranked_candidates = filtered_candidates
                else:
                    return unique_candidates[:top_k]
            return reranked_candidates
        
        return unique_candidates[:top_k]

    def _restore_bm25_index(self) -> None:
        """Keep BM25 available when ChromaDB survives a backend restart."""
        if bm25_retriever.get_total_chunks() == 0 and vector_store.get_total_chunks() > 0:
            bm25_retriever.restore_from_chunks(vector_store.get_all_chunks())

    def _retrieve_variations(
        self,
        queries: List[str],
        candidate_k: int,
        mode: Literal["hybrid", "vector", "bm25"],
        rrf_k: int,
    ) -> List[Dict[str, Any]]:
        """Retrieve and merge results for all rewritten query variants."""
        merged: Dict[str, Dict[str, Any]] = {}
        rank_scores: Dict[str, float] = {}
        for query in queries:
            if mode == "vector":
                results = vector_store.similarity_search(query, top_k=candidate_k)
            elif mode == "bm25":
                bm25_results = bm25_retriever.search(query, top_k=candidate_k)
                results = [
                    {
                        "text": item["text"],
                        "source": item["source"],
                        "chunk_index": item["chunk_index"],
                        "similarity_score": item["score"],
                    }
                    for item in bm25_results
                ]
                # Some documents contain a phone number or email value without
                # the label "phone" or "email". Use dense similarity only when
                # the original lexical query has no BM25 evidence.
                if not results and query == queries[0]:
                    results = vector_store.similarity_search(query, top_k=candidate_k)
            else:
                results = self._reciprocal_rank_fusion(query, top_k=candidate_k, rrf_k=rrf_k)

            for rank, item in enumerate(results, start=1):
                key = self._hash_text(item["text"])
                merged.setdefault(key, dict(item))
                original_query_bonus = 1.0 if query == queries[0] else 0.0
                rank_scores[key] = rank_scores.get(key, 0.0) + original_query_bonus + (1.0 / (rrf_k + rank))

        ordered = sorted(merged, key=lambda key: rank_scores[key], reverse=True)
        output = []
        for key in ordered[:candidate_k]:
            item = dict(merged[key])
            if mode == "hybrid":
                item["rrf_score"] = round(rank_scores[key], 5)
            output.append(item)
        return output

    def _reciprocal_rank_fusion(
        self, 
        query: str, 
        top_k: int = 10, 
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Perform Reciprocal Rank Fusion (RRF) on candidates from Vector and BM25 search.
        """
        vector_results = vector_store.similarity_search(query, top_k=top_k)
        bm25_results = bm25_retriever.search(query, top_k=top_k)

        rrf_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(vector_results, start=1):
            key = self._hash_text(item["text"])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))
            chunk_lookup[key] = item

        for rank, item in enumerate(bm25_results, start=1):
            key = self._hash_text(item["text"])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))
            if key not in chunk_lookup:
                chunk_lookup[key] = {
                    "text": item["text"],
                    "source": item["source"],
                    "chunk_index": item["chunk_index"],
                    "similarity_score": item["score"]
                }

        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        fused_results = []
        for key in sorted_keys[:top_k]:
            item = dict(chunk_lookup[key])
            item["rrf_score"] = round(rrf_scores[key], 5)
            fused_results.append(item)

        return fused_results

# Global singleton instance
hybrid_retriever = HybridRetriever()
