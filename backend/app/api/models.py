from typing import List, Optional, Literal, Union, Any
from pydantic import BaseModel, Field

class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int
    total_indexed_chunks: int

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question or prompt")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of context passages to retrieve")
    retrieval_mode: Literal["hybrid", "vector", "bm25"] = Field(
        default="hybrid", 
        description="Retrieval strategy: 'hybrid', 'vector', or 'bm25'"
    )
    use_reranker: bool = Field(
        default=True,
        description="Whether to run Cross-Encoder reranking post-retrieval stage"
    )

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question about the uploaded documents")
    search_mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    use_reranker: bool = True
    top_k: int = Field(default=10, ge=1, le=20)

class RetrievedChunkSource(BaseModel):
    text: str
    source: str
    chunk_index: int
    similarity_score: float
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None

class ChatResponse(BaseModel):
    question: str
    answer: str
    retrieval_mode: str
    use_reranker: bool
    sources: List[RetrievedChunkSource]

class EvaluationRequest(BaseModel):
    question: str
    answer: str
    retrieved_context: List[str]
    ground_truth: Optional[str] = None

class EvaluationResponse(BaseModel):
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    context_recall: Union[float, str]
    overall_score: float

class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    embedding_model: str
    document_chunks_indexed: int
