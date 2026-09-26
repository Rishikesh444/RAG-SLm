import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.core.config import settings
from app.api.models import (
    HealthResponse,
    DocumentUploadResponse,
    ChatRequest,
    ChatResponse,
    RetrievedChunkSource,
    EvaluationRequest,
    EvaluationResponse,
    QueryRequest
)
from app.services.extractor import extract_text_from_file, TextExtractionError
from app.services.chunker import chunk_text
from app.services.vector_store import vector_store
from app.services.bm25_retriever import bm25_retriever
from app.services.hybrid_retriever import hybrid_retriever
from app.services.llm import generate_answer
from app.services.query_preprocessor import preprocess_query
from app.evaluation.evaluator import evaluator

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint returning system status and indexed document count."""
    return HealthResponse(
        status="ok",
        llm_provider=settings.LLM_PROVIDER,
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        document_chunks_indexed=vector_store.get_total_chunks()
    )

@router.post("/documents/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
@router.post("/api/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, DOCX), extract text, chunk it, and index into Vector Store & BM25 Index.
    """
    filename = file.filename
    extension = os.path.splitext(filename)[1].lower()
    if extension not in [".txt", ".pdf", ".docx"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{extension}'. Allowed formats: .txt, .pdf, .docx"
        )

    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

    try:
        text = extract_text_from_file(file_path, filename)
    except TextExtractionError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

    chunks = chunk_text(text, source_name=filename)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document contained no text content to process."
        )

    indexed_count = vector_store.add_chunks(chunks)
    bm25_retriever.add_chunks(chunks)

    return DocumentUploadResponse(
        message=f"Successfully uploaded and indexed '{filename}'.",
        filename=filename,
        chunks_created=indexed_count,
        total_indexed_chunks=vector_store.get_total_chunks()
    )

@router.post("/chat", response_model=ChatResponse)
def chat_with_documents(request: ChatRequest):
    """
    Query uploaded documents using Vector, BM25, or Hybrid retrieval with optional Cross-Encoder reranking.
    """
    if vector_store.get_total_chunks() == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been uploaded yet. Please upload at least one document before asking questions."
        )

    # 1. Rewrite and expand the query before retrieval.
    query_variations = preprocess_query(request.question)

    # 2. Multi-query retrieval + optional Reranker
    retrieved_chunks = hybrid_retriever.retrieve(
        query=request.question, 
        top_k=request.top_k, 
        mode=request.retrieval_mode,
        use_reranker=request.use_reranker,
        query_variations=query_variations,
    )

    # 3. Generate grounded answer
    answer = generate_answer(
        query=request.question,
        retrieved_chunks=retrieved_chunks,
        query_variations=query_variations,
    )

    # 4. Format citations with vector, RRF, and rerank scores
    sources = [
        RetrievedChunkSource(
            text=c["text"],
            source=c["source"],
            chunk_index=c["chunk_index"],
            similarity_score=c.get("similarity_score", 0.0),
            rrf_score=c.get("rrf_score"),
            rerank_score=c.get("rerank_score")
        )
        for c in retrieved_chunks
    ]

    return ChatResponse(
        question=request.question,
        answer=answer,
        retrieval_mode=request.retrieval_mode,
        use_reranker=request.use_reranker,
        sources=sources
    )

@router.post("/api/query", response_model=ChatResponse)
def query_documents(request: QueryRequest):
    """Public API contract for querying the document research assistant."""
    return chat_with_documents(ChatRequest(
        question=request.query,
        top_k=request.top_k,
        retrieval_mode=request.search_mode,
        use_reranker=request.use_reranker,
    ))

@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate_rag(request: EvaluationRequest):
    """
    Evaluate RAG pipeline output across Faithfulness, Relevance, and Context Recall metrics.
    """
    chunks = [{"text": text} for text in request.retrieved_context]
    metrics = evaluator.evaluate(
        question=request.question,
        answer=request.answer,
        retrieved_chunks=chunks,
        ground_truth=request.ground_truth
    )
    return EvaluationResponse(**metrics)

@router.delete("/documents", status_code=status.HTTP_200_OK)
def clear_documents():
    """Clear all stored documents and vectors."""
    vector_store.clear_all()
    bm25_retriever.clear_all()
    return {"message": "All documents, vector embeddings, and BM25 indices have been cleared."}
