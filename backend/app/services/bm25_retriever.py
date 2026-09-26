import re
from typing import List, Dict, Any
from rank_bm25 import BM25Plus

class BM25RetrieverManager:
    """
    Manages in-memory BM25 lexical keyword search over document text chunks.
    Uses BM25Plus algorithm which prevents zero/negative IDF scores on small corpuses.
    """
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Plus = None

    def tokenize(self, text: str) -> List[str]:
        """Lowercase word tokenization using regex."""
        return re.findall(r'\w+', text.lower())

    def _index_tokens(self, text: str) -> List[str]:
        """Tokenize text and add lexical labels for unlabeled contact values."""
        tokens = self.tokenize(text)
        if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text):
            tokens.extend(["email", "contact"])
        if re.search(r"(?:\+?\d[\d\s().-]{7,}\d)", text):
            tokens.extend(["phone", "mobile", "number", "contact"])
        return tokens

    def _meaningful_tokens(self, text: str) -> set[str]:
        """Return query terms that carry lexical meaning for retrieval."""
        stopwords = {
            "a", "about", "an", "and", "are", "can", "could", "did", "do", "does",
            "for", "from", "how", "i", "in", "is", "it", "me", "of", "on", "or",
            "please", "tell", "that", "the", "this", "to", "was", "what", "which",
            "who", "with", "would", "you"
        }
        return {
            token for token in self.tokenize(text)
            if token not in stopwords and (len(token) > 1 or any(char.isdigit() for char in token))
        }

    def add_chunks(self, new_chunks: List[Dict[str, Any]]) -> int:
        """
        Add new document chunks to BM25 index and rebuild index.
        """
        if not new_chunks:
            return 0

        for chunk in new_chunks:
            tokens = self._index_tokens(chunk["text"])
            self.chunks.append(chunk)
            self.corpus_tokens.append(tokens)

        # Rebuild BM25Plus index over all corpus tokens
        if self.corpus_tokens:
            self.bm25 = BM25Plus(self.corpus_tokens)

        return len(new_chunks)

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Perform BM25 lexical keyword search for query against indexed corpus.
        
        Returns list of matched chunk dicts sorted by BM25 score.
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        meaningful_query_tokens = self._meaningful_tokens(query)
        if not meaningful_query_tokens:
            return []

        # Calculate BM25Plus scores for all document chunks
        scores = self.bm25.get_scores(query_tokens)

        # Pair each chunk with its BM25 score
        scored_chunks = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            chunk_tokens = set(self.tokenize(chunk["text"]))
            if score <= 0 or not meaningful_query_tokens.intersection(chunk_tokens):
                continue

            scored_chunks.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "source": chunk["metadata"]["source"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "score": round(float(score), 4)
            })

        # Sort descending by BM25 score
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    def clear_all(self) -> None:
        """Clear all stored chunks and reset BM25 index."""
        self.chunks = []
        self.corpus_tokens = []
        self.bm25 = None

    def get_total_chunks(self) -> int:
        """Return total number of chunks currently indexed in BM25."""
        return len(self.chunks)

    def restore_from_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Restore the lexical index after a process restart without duplicating data."""
        if self.chunks or not chunks:
            return len(self.chunks)
        self.chunks = list(chunks)
        self.corpus_tokens = [self._index_tokens(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Plus(self.corpus_tokens)
        return len(self.chunks)

# Global singleton instance
bm25_retriever = BM25RetrieverManager()
