from pathlib import Path
from contextlib import asynccontextmanager
import os
import re
import hmac

import numpy as np
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Header
)
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------
# CONFIGURATION
# --------------------------------

API_KEY = os.getenv("API_KEY", "week6-local-secret-key")

model = None
document_chunks = []
chunk_embeddings = None

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=50
)


# --------------------------------
# API KEY VALIDATION
# --------------------------------

def verify_api_key(x_api_key: str):

    if not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )


# --------------------------------
# EMBEDDING FUNCTION
# --------------------------------

def generate_embeddings(texts):

    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )


# --------------------------------
# DOCUMENT LOADING
# --------------------------------

def load_documents():

    global model
    global document_chunks
    global chunk_embeddings

    print("Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    task2_folder = (
        Path(__file__).resolve().parent.parent / "Task2"
    )

    if not task2_folder.exists():
        raise FileNotFoundError("Task2 folder not found!")

    document_chunks.clear()

    for file_path in task2_folder.glob("*.txt"):

        content = file_path.read_text(encoding="utf-8")

        sections = re.split(
            r"(?=Section \d+:)",
            content
        )

        for section in sections:

            if not section.strip():
                continue

            heading_match = re.search(
                r"Section \d+:.*",
                section
            )

            heading = (
                heading_match.group(0)
                if heading_match
                else "General"
            )

            for index, chunk in enumerate(
                splitter.split_text(section)
            ):

                document_chunks.append({
                    "source": file_path.name,
                    "page": index + 1,
                    "section": heading,
                    "doc_type": "TXT",
                    "text": chunk
                })

    if not document_chunks:
        raise RuntimeError("No documents found in Task2!")

    texts = [
        item["text"] for item in document_chunks
    ]

    chunk_embeddings = generate_embeddings(texts)

    print("Documents loaded:", len(document_chunks))
    print("Embeddings generated successfully!")


# --------------------------------
# APPLICATION STARTUP
# --------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    load_documents()

    yield


app = FastAPI(
    title="Semantic Document Search API",
    description="Search and upload documents using embeddings",
    version="1.0.0",
    lifespan=lifespan
)


# --------------------------------
# REQUEST MODEL
# --------------------------------

class SearchRequest(BaseModel):

    query: str = Field(min_length=1)

    top_k: int = Field(
        default=5,
        ge=1,
        le=20
    )

    doc_type: str | None = None

    min_score: float | None = Field(
        default=None,
        ge=0,
        le=1
    )


# --------------------------------
# HOME ENDPOINT
# --------------------------------

@app.get("/")
def home():

    return {
        "message": "Semantic Document Search API is running!"
    }


# --------------------------------
# HEALTH ENDPOINT
# --------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "documents_loaded": len(document_chunks),
        "model_loaded": model is not None
    }


# --------------------------------
# SEARCH ENDPOINT
# --------------------------------

@app.post("/search")
def search(
    request: SearchRequest,
    x_api_key: str = Header(...)
):

    verify_api_key(x_api_key)

    if model is None or chunk_embeddings is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready"
        )

    query_embedding = generate_embeddings(
        [request.query]
    )[0]

    scores = np.dot(
        chunk_embeddings,
        query_embedding
    )

    results = []

    for index, score in enumerate(scores):

        item = document_chunks[index]

        # Document type filter
        if request.doc_type:

            if item["doc_type"].upper() != request.doc_type.upper():
                continue

        # Minimum similarity score
        if request.min_score is not None:

            if float(score) < request.min_score:
                continue

        results.append({
            "source": item["source"],
            "page": item["page"],
            "section": item["section"],
            "doc_type": item["doc_type"],
            "score": round(float(score), 4),
            "text": item["text"]
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    results = results[:request.top_k]

    return {
        "query": request.query,
        "top_k": request.top_k,
        "total_results": len(results),
        "results": results
    }


# --------------------------------
# DOCUMENT UPLOAD ENDPOINT
# --------------------------------

@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    x_api_key: str = Header(...)
):

    global document_chunks
    global chunk_embeddings

    verify_api_key(x_api_key)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready"
        )

    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only TXT files are supported"
        )

    try:
        content = (await file.read()).decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must use UTF-8 encoding"
        )

    if not content.strip():
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    new_chunks = splitter.split_text(content)

    if not new_chunks:
        raise HTTPException(
            status_code=400,
            detail="No text chunks found"
        )

    new_items = []

    for index, chunk in enumerate(new_chunks):

        new_items.append({
            "source": file.filename,
            "page": index + 1,
            "section": "Uploaded Document",
            "doc_type": "TXT",
            "text": chunk
        })

    new_embeddings = generate_embeddings(
        [item["text"] for item in new_items]
    )

    document_chunks.extend(new_items)

    if chunk_embeddings is None:
        chunk_embeddings = new_embeddings
    else:
        chunk_embeddings = np.vstack([
            chunk_embeddings,
            new_embeddings
        ])

    return {
        "message": "Document uploaded and indexed successfully",
        "filename": file.filename,
        "chunks_added": len(new_items),
        "total_chunks": len(document_chunks)
    }


print("FastAPI application configured successfully!")