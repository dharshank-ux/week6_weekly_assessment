# Task 3 – Vector Databases

## Overview
This task demonstrates Approximate Nearest Neighbor (ANN), HNSW indexing, and vector database operations.

## 1. ANN and HNSW
- Generated 50,000 random vectors.
- Compared exact search with HNSW approximate search.
- Measured build time and search time.
- Calculated Recall@10.

## 2. Qdrant Vector Database
- Loaded document chunks from Task2.
- Generated embeddings using all-MiniLM-L6-v2.
- Stored vectors and metadata in Qdrant.
- Configured HNSW indexing.

## 3. Metadata Filtering
Tested filtered vector search using:
- Document type
- Page number

## 4. Duplicate Prevention
Used stable UUID5 IDs.
Repeated upsert operations to verify that the point count does not increase.

## 5. Delete Operation
Tested deleting chunks using a source-file filter.

## 6. Database Comparison

### Qdrant
- Dedicated vector database
- HNSW support
- Metadata filtering
- Upsert and delete operations

### PostgreSQL with pgvector
- Vector search with SQL
- HNSW and IVFFlat indexes
- Relational data support

## Run Files
bash
py task3_ann_hnsw.py
py task3_qdrant.py
py task3_comparison.py


## Note
Qdrant was tested using an in-memory database. PostgreSQL with pgvector was compared conceptually, not executed.