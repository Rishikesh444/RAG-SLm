import os
import chromadb
from typing import List, Dict, Any
from app.core.config import settings

class VectorStoreManager:
    """
    Manages persistent vector storage and similarity search using ChromaDB
    and ONNX-accelerated embeddings (all-MiniLM-L6-v2).
    Lazy-initializes the embedding function and collection to avoid startup
    crashes when ONNX DLLs are blocked by Application Control policy.
    """
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
        self._embedding_fn = None
        self._collection = None

    def _get_embedding_fn(self):
        """Lazy-load the ONNX embedding function on first use."""
        if self._embedding_fn is None:
            from chromadb.utils import embedding_functions
            self._embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        return self._embedding_fn

    def _get_collection(self):
        """Lazy-initialize ChromaDB collection on first use."""
        if self._collection is None:
            self._collection = self.chroma_client.get_or_create_collection(
                name="document_chunks",
                embedding_function=self._get_embedding_fn(),
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    @property
    def collection(self):
        return self._get_collection()

    @collection.setter
    def collection(self, value):
        self._collection = value

    @property
    def embedding_fn(self):
        return self._get_embedding_fn()

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Store text chunks and metadata in ChromaDB."""
        if not chunks:
            return 0

        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            return len(chunks)
        except Exception as e:
            # Fallback re-get collection if connection handle stale
            self.collection = self.chroma_client.get_or_create_collection(
                name="document_chunks",
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
            return len(chunks)

    def similarity_search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Perform vector similarity search against indexed document chunks."""
        try:
            total_docs = self.collection.count()
        except Exception:
            return []

        if total_docs == 0:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, total_docs),
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            return []

        formatted_results = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for doc, meta, dist in zip(docs, metas, distances):
                similarity_score = round(1.0 - dist, 4) if dist is not None else 0.0
                formatted_results.append({
                    "text": doc,
                    "source": meta.get("source", "Unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "similarity_score": similarity_score
                })

        return formatted_results

    def get_total_chunks(self) -> int:
        """Return total number of chunks currently stored in ChromaDB."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Return stored chunks so the in-memory lexical index can be restored."""
        try:
            results = self.collection.get(include=["documents", "metadatas"])
        except Exception:
            return []

        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        chunks = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            chunks.append({
                "id": f"restored_{index}",
                "text": text,
                "metadata": metadata,
            })
        return chunks

    def clear_all(self) -> None:
        """Clear all stored vectors and documents from ChromaDB collection."""
        try:
            self.chroma_client.delete_collection("document_chunks")
        except Exception:
            pass
            
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_chunks",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

# Global singleton instance
vector_store = VectorStoreManager()
