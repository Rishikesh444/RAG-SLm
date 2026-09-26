# Intelligent Document Research Assistant — Hybrid RAG Chatbot

An enterprise-grade, production-quality Retrieval-Augmented Generation (RAG) platform built with **Python**, **FastAPI**, **React**, **ChromaDB**, **BM25**, **FlashRank Cross-Encoder Reranking**, and **Google Gemini / OpenAI**.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-emerald)
![React](https://img.shields.io/badge/React-18-cyan)

---

## Architecture Overview

```
+-----------------------------------------------------------------------------------+
|                                 FRONTEND                                          |
|                React 18 + Vite + Custom Modern Dark UI                            |
|          - Document Upload Dropzone (.pdf, .docx, .txt)                           |
|          - Interactive RAG Tuning (Hybrid / Vector / BM25, Top-K, Reranker)      |
|          - Chat thread with Citation Cards & RAG Quality Evaluator                |
+-----------------------------------------------------------------------------------+
                                          | HTTP / REST
                                          v
+-----------------------------------------------------------------------------------+
|                                 FASTAPI BACKEND                                   |
|                                                                                   |
|  +----------------------+   +-----------------------+   +----------------------+  |
|  | Document Ingestion   |   | Hybrid Retrieval      |   | LLM Generation       |  |
|  | - PDF (PyPDF)        |   | - Vector (ChromaDB)   |   | - Grounded Prompting |  |
|  | - DOCX (python-docx) |   | - BM25 (BM25Plus)     |   | - Gemini / OpenAI    |  |
|  | - Recursive Chunker  |   | - RRF Rank Fusion     |   | - Source Citations   |  |
|  +----------------------+   | - Cross Reranker      |   +----------------------+  |
|                             +-----------------------+                             |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                 STORAGE LAYER                                     |
|           ChromaDB (Vector Store)  +  In-Memory BM25 Lexical Index              |
+-----------------------------------------------------------------------------------+
```

---

## Core Features & Multi-Stage Pipeline

### 1. Document Ingestion & Text Extraction
* Supports parsing `.pdf` (PyPDF), `.docx` (python-docx), and `.txt` files into clean raw text.
* Uses `RecursiveCharacterTextSplitter` to divide continuous documents into overlapping semantic chunks (500 characters, 100 character overlap).

### 2. Multi-Modal Hybrid Retrieval
* **Dense Vector Search**: Embeds chunks using an ONNX-compiled `all-MiniLM-L6-v2` transformer model (384 dimensions) persisted in **ChromaDB** with cosine distance index.
* **Sparse Lexical Search**: Indexes tokenized document terms into an in-memory **BM25Plus** engine to capture exact keywords, technical codes, and proper nouns without zero-IDF penalties.
* **Reciprocal Rank Fusion (RRF)**: Merges ranked candidate lists from both retrievers using score-agnostic rank aggregation:
  $$RRF\_Score(d) = \sum_{m \in \{Vector, BM25\}} \frac{1}{60 + r_m(d)}$$

### 3. Cross-Encoder Reranking
* Candidate passages retrieved from hybrid search pass through a **Cross-Encoder Neural Network** (`ms-marco-TinyBERT-L-2-v2` via **FlashRank ONNX**).
* Cross-encoders process the query and passage simultaneously through self-attention layers, providing far greater relevance precision than vector similarity alone.

### 4. Grounded Synthesis & Citations
* Context-aware prompt design enforces strict grounding—preventing hallucinations when uploaded documents do not contain sufficient information.
* Every answer includes structured source metadata (document filename, chunk index, vector similarity score, RRF score, and reranker score).

### 5. RAG Quality Evaluation
* Includes a quantitative evaluation suite measuring:
  * **Faithfulness**: Claim grounding in retrieved context.
  * **Answer Relevance**: Response alignment with user question.
  * **Context Relevance**: Information density of retrieved chunks.
  * **Context Recall**: Coverage of ground truth targets.

---

## Project Structure

```
rag-project/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── endpoints.py     # REST API routes (/documents/upload, /chat, /evaluate, /health)
│   │   │   └── models.py        # Pydantic data schemas
│   │   ├── core/
│   │   │   └── config.py        # Environment settings & configuration
│   │   ├── evaluation/
│   │   │   └── evaluator.py     # RAG metric evaluation suite
│   │   ├── services/
│   │   │   ├── extractor.py     # Text extraction (PDF, DOCX, TXT)
│   │   │   ├── chunker.py       # Recursive character text chunking
│   │   │   ├── vector_store.py  # ChromaDB vector manager (ONNX MiniLM)
│   │   │   ├── bm25_retriever.py# BM25Plus keyword retriever
│   │   │   ├── reranker.py      # FlashRank Cross-Encoder reranker
│   │   │   ├── hybrid_retriever.py # Reciprocal Rank Fusion engine
│   │   │   └── llm.py           # Gemini / OpenAI synthesis
│   ├── storage/                 # Persistent vector store & uploads
│   ├── tests/
│   │   ├── test_phase1.py       # Basic RAG test suite
│   │   └── test_phase2.py       # Hybrid retrieval test suite
│   ├── .env.example
│   ├── main.py                  # FastAPI application entrypoint
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # React single-page UI
│   │   ├── index.css            # Custom CSS design system
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## Getting Started

### Prerequisites
* **Python 3.11+**
* **Node.js 18+** & **npm**

---

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   Copy `.env.example` to `.env` and add your LLM key:
   ```env
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

5. Run verification tests:
   ```bash
   python tests/test_phase1.py
   python tests/test_phase2.py
   ```

6. Start the FastAPI server:
   ```bash
   python main.py
   ```
   FastAPI server runs at `http://127.0.0.1:8000` (Swagger UI at `http://127.0.0.1:8000/docs`).

---

### Frontend Setup

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   Open `http://localhost:5173` in your browser.

---

## API Documentation Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health status and indexed chunk metrics. |
| `POST` | `/documents/upload` | Upload `.pdf`, `.docx`, or `.txt` file and index into ChromaDB & BM25. |
| `POST` | `/chat` | Query RAG pipeline (`question`, `top_k`, `retrieval_mode`, `use_reranker`). |
| `POST` | `/evaluate` | Evaluate RAG response across Faithfulness, Relevance, and Recall. |
| `DELETE` | `/documents` | Clear all stored vector embeddings and BM25 indices. |

---

## Technical Interview Questions & Architecture Answers

1. **Why use Hybrid Search instead of Dense Vector Search alone?**  
   *Answer*: Vector search measures semantic similarity in embedding space, which works well for conceptual queries. However, it frequently misses exact string tokens such as product serial numbers, error codes (`ERR_9901`), or acronyms. BM25 provides term-frequency/inverse-document-frequency precision for exact keywords.

2. **What problem does Reciprocal Rank Fusion (RRF) solve?**  
   *Answer*: Cosine similarity scores ($0 \dots 1$) and BM25 scores ($0 \dots \infty$) are on completely different scales. Instead of attempting brittle score normalization, RRF aggregates documents based on their ordinal rank positions ($1 / (k + rank)$), ensuring scale-free consensus.

3. **Why use a Cross-Encoder Reranker post-retrieval stage?**  
   *Answer*: Vector embeddings rely on bi-encoders that process queries and documents independently. A Cross-Encoder feeds the query and passage together through a Transformer model, allowing cross-attention between every query word and passage word. Running Cross-Encoder on only top-10 candidate passages gives maximum relevance accuracy with minimal latency cost (~20ms).
