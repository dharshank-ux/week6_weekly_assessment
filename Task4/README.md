Week 6 – Task 4: FastAPI Semantic Search

Project Overview

A semantic document search API built using FastAPI and Sentence Transformers.

Technologies Used

- Python 3.14.7
- FastAPI
- Uvicorn
- Sentence Transformers
- all-MiniLM-L6-v2
- NumPy
- Pytest

Features

1. Semantic document search.
2. POST "/search" endpoint.
3. GET "/health" endpoint.
4. POST "/documents" upload endpoint.
5. API key authentication.
6. Top-K search results.
7. Document type filtering.
8. Minimum similarity score filtering.
9. Search evaluation using Hit@1, Hit@3, Hit@5 and MRR.
10. Automated testing with Pytest.

API Endpoints

Method| Endpoint| Purpose
GET| "/"| Home
GET| "/health"| Health check
POST| "/search"| Semantic search
POST| "/documents"| Upload TXT document

Evaluation

- 15 evaluation queries.
- Hit@1, Hit@3 and Hit@5.
- Mean Reciprocal Rank (MRR).
- Chunk size 256 vs 512 comparison.

See "configuration_comparison.md" for configuration details and actual evaluation metrics.

Testing

Run:

py -m pytest test_main.py -v

Six automated tests are included.

Run the Application

py -m uvicorn main:app --reload

API URL: "http://127.0.0.1:8000"

Authentication

The API uses an X-API-Key header for protected endpoints.

Local testing key: "week6-local-secret-key"

For production, configure a secure API key using the API_KEY environment variable.

Notes

- The application loads TXT documents from the Task2 folder.
- Uploaded TXT documents are indexed in memory.
- TXT page metadata is represented by synthetic chunk indices.
- The local embedding model is used without requiring an external API key.