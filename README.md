# LLM Chat System — Production-Grade RAG

A production-ready Retrieval-Augmented Generation (RAG) chat system
built with direct SDK integrations. No LangChain. No wrappers.

---

## Architecture

Upload → Validate → Hash Registry Check
↓
Save to S3
↓
Unstructured.io Parsing
(Tables, Lists, Code, Narrative)
↓
Element-aware Chunking
(Never splits tables or code)
↓
OpenAI Embeddings (text-embedding-3-small)
↓
Qdrant Vector DB
↓
Query → Exact Cache → Semantic Cache → Qdrant Search
↓
Build Context + Prompt
↓
OpenAI gpt-4o-mini
↓
Return Answer + Sources
(with S3 presigned URLs)
↓
Langfuse Trace + RAGAS Evals


---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python + FastAPI |
| LLM | OpenAI gpt-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | Qdrant |
| Cache | Redis + redisvl (exact + semantic) |
| Object Storage | AWS S3 (private + presigned URLs) |
| Document Parsing | Unstructured.io (element-aware) |
| Observability | Langfuse |
| Evals | RAGAS (faithfulness, relevancy, precision) |

---

## Key Features

- **Element-aware chunking** — tables, code blocks, and lists are
  never split mid-structure. Each element type is cleaned and
  chunked with type-specific rules.

- **Content-hash document registry** — documents are identified
  by SHA256 of their content, not their filename. Renames,
  duplicates, and content updates are all handled correctly.
  Stale Qdrant chunks are cleaned up automatically on update.

- **Two-layer caching** — exact match (SHA256 key in Redis) +
  semantic match (cosine similarity via redisvl). Cache hit type
  is returned with every response.

- **Full observability** — every ingestion and query is traced in
  Langfuse with latency, token counts, USD cost, and RAGAS scores.

- **No LangChain** — all integrations are direct SDK calls
  (openai, qdrant-client, boto3, langfuse). Every step is
  explicit and debuggable.

---

## Project Structure

llm-chat-system/
├── ingestion/
│ ├── validator.py # File type + size validation
│ ├── extractor.py # Unstructured.io element parsing
│ ├── cleaner.py # Per-element type cleaning
│ ├── chunker.py # Element-aware hybrid chunking
│ └── pipeline.py # Ingestion orchestration
├── storage/
│ ├── object_store.py # S3 upload, presigned URL, delete
│ └── document_registry.py # SHA256 hash registry (Redis)
├── query/
│ ├── embedder.py # OpenAI embeddings (shared)
│ ├── retriever.py # Qdrant search + access filter
│ ├── prompt_builder.py # Prompt assembly
│ └── inference.py # Query orchestration
├── cache/
│ └── manager.py # Exact + semantic cache
├── observability/
│ ├── langfuse_client.py # Tracing setup
│ └── cost_tracker.py # Token → USD cost
├── evals/
│ ├── ragas_runner.py # RAGAS evaluation
│ └── score_logger.py # Async score upload to Langfuse
├── api/
│ └── routes.py # FastAPI endpoints
├── docker-compose.yml
├── requirements.txt
└── main.py


---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/RahulJ0hn/llm-chat-system
cd llm-chat-system
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your keys:
# OPENAI_API_KEY, AWS credentials, Langfuse keys
```

### 3. Start infrastructure

```bash
docker compose up -d
# Starts Qdrant (port 6333) and Redis (port 6379)
```

### 4. Run the server

```bash
uvicorn main:app --reload
```

---

## API Reference

### POST /ingest
Upload a document (PDF, DOCX, TXT).

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@contract.pdf"
```

Response:
```json
{
  "status": "created",
  "doc_name": "contract.pdf",
  "doc_hash": "abc123...",
  "chunks_stored": 42,
  "element_counts": {
    "narrative": 35,
    "table": 4,
    "list": 3,
    "code": 0
  },
  "ingestion_ms": 1842.5,
  "skipped": false
}
```

### POST /query
Ask a question against ingested documents.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the termination clause?", "user_role": "public"}'
```

Response:
```json
{
  "answer": "The termination clause requires 90 days written notice...",
  "sources": [
    {
      "doc_name": "contract.pdf",
      "section": "Termination",
      "page_number": 7,
      "element_type": "narrative",
      "score": 0.9412,
      "download_url": "https://s3.ap-south-1.amazonaws.com/..."
    }
  ],
  "cache_hit": false,
  "cache_type": "none",
  "eval_scores": {
    "faithfulness": 0.9800,
    "answer_relevancy": 0.9500,
    "context_precision": 0.9200
  }
}
```

### DELETE /document/{doc_hash}
Delete a document and all its chunks.

```bash
curl -X DELETE http://localhost:8000/document/abc123...
```

### GET /documents
List all ingested documents.

```bash
curl http://localhost:8000/documents
```

### GET /cache/stats
Cache hit/miss statistics.

```bash
curl http://localhost:8000/cache/stats
```

### DELETE /cache/flush
Clear all cached responses.

```bash
curl -X DELETE http://localhost:8000/cache/flush
```

### GET /health
Infrastructure health check.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "qdrant": "connected",
  "redis": "connected",
  "s3": "connected"
}
```

---

## Environment Variables

| Variable | Description |
|---|---|
| OPENAI_API_KEY | OpenAI API key |
| QDRANT_URL | Qdrant URL (default: http://localhost:6333) |
| QDRANT_COLLECTION | Collection name (default: documents) |
| REDIS_URL | Redis URL (default: redis://localhost:6379) |
| EMBEDDING_MODEL | OpenAI embedding model |
| EMBEDDING_DIM | Embedding dimensions (1536) |
| LLM_MODEL | OpenAI chat model |
| LLM_TEMPERATURE | LLM temperature (0.2) |
| CACHE_TTL | Cache TTL in seconds (3600) |
| CACHE_SIMILARITY_THRESHOLD | Semantic cache threshold (0.92) |
| AWS_ACCESS_KEY_ID | AWS access key |
| AWS_SECRET_ACCESS_KEY | AWS secret key |
| AWS_REGION | AWS region (ap-south-1) |
| S3_BUCKET_NAME | S3 bucket name |
| S3_PRESIGN_EXPIRY | Presigned URL expiry in seconds |
| LANGFUSE_PUBLIC_KEY | Langfuse public key |
| LANGFUSE_SECRET_KEY | Langfuse secret key |
| LANGFUSE_HOST | Langfuse host URL |
| RUN_EVALS | Enable RAGAS evals (true/false) |
| EVAL_ASYNC | Run evals async (true/false) |

---

## Roadmap

- [ ] OCR for scanned PDFs (Tesseract / AWS Textract)
- [ ] Page-level hash comparison for partial doc updates
- [ ] Streaming response endpoint
- [ ] Reranker before LLM call (Cohere Rerank)
- [ ] Hybrid search — dense + sparse BM25
- [ ] User authentication + JWT role management
- [ ] GraphRAG for cross-document relationship queries

---

## Author

Rahul John — Agentic AI Engineer
GitHub: github.com/RahulJ0hn
