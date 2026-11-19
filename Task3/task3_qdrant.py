from pathlib import Path
from uuid import uuid5, NAMESPACE_DNS

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

# Load documents from Task2
task2_folder = Path(__file__).resolve().parent.parent / "Task2"

documents = []
for file_path in task2_folder.glob("*.txt"):
    documents.append((file_path.name, file_path.read_text(encoding="utf-8")))

if not documents:
    raise RuntimeError("No TXT documents found in Task2 folder.")

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)

chunks = []

for source, content in documents:
    for index, chunk_text in enumerate(splitter.split_text(content)):
        chunks.append({
            "source": source,
            "text": chunk_text,
            "page": index + 1,  # Synthetic page number for TXT chunks
            "doc_type": "TXT",
            "chunk_index": index
        })

print("Documents loaded:", len(documents))
print("Chunks created:", len(chunks))

# Load local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts, normalize_embeddings=True)

# Create in-memory Qdrant database
client = QdrantClient(":memory:")
collection_name = "week6_documents"

client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE
    ),
    hnsw_config=models.HnswConfigDiff(
        m=16,
        ef_construct=100
    )
)

# Create payload indexes for filtering
client.create_payload_index(
    collection_name=collection_name,
    field_name="doc_type",
    field_schema=models.PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name=collection_name,
    field_name="page",
    field_schema=models.PayloadSchemaType.INTEGER
)

client.create_payload_index(
    collection_name=collection_name,
    field_name="source",
    field_schema=models.PayloadSchemaType.KEYWORD
)

# Prepare points with stable IDs
points = []

for chunk, vector in zip(chunks, embeddings):
    stable_id = str(
        uuid5(
            NAMESPACE_DNS,
            f'{chunk["source"]}:{chunk["chunk_index"]}:{chunk["text"]}'
        )
    )

    points.append(
        models.PointStruct(
            id=stable_id,
            vector=vector.tolist(),
            payload=chunk
        )
    )

# First upsert
client.upsert(
    collection_name=collection_name,
    points=points,
    wait=True
)

count1 = client.count(
    collection_name=collection_name,
    exact=True
).count

print("\nFirst indexing count:", count1)

# Re-upsert same points to test duplicate prevention
client.upsert(
    collection_name=collection_name,
    points=points,
    wait=True
)

count2 = client.count(
    collection_name=collection_name,
    exact=True
).count

print("After second indexing count:", count2)
print("Duplicate prevention:", count1 == count2)

# Filtered search: doc_type TXT and page >= 3
query_vector = model.encode(
    "What are the company security rules?",
    normalize_embeddings=True
).tolist()

search_result = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(value="TXT")
            ),
            models.FieldCondition(
                key="page",
                range=models.Range(gte=3)
            )
        ]
    ),
    limit=5
).points

print("\nFiltered search results:")
for result in search_result:
    print(
        result.payload["source"],
        "Page:", result.payload["page"],
        "Score:", round(result.score, 4)
    )

# Delete chunks belonging to one source
delete_source = "policy.txt"

client.delete(
    collection_name=collection_name,
    points_selector=models.FilterSelector(
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=delete_source)
                )
            ]
        )
    ),
    wait=True
)

final_count = client.count(
    collection_name=collection_name,
    exact=True
).count

print("\nDeleted source:", delete_source)
print("Count before deletion:", count2)
print("Count after deletion:", final_count)

# Collection information
info = client.get_collection(collection_name)

print("\nQdrant Collection Information")
print("Status:", info.status)
print("Points remaining:", info.points_count)
print("HNSW configuration:", info.config.hnsw_config)

print("\nQdrant experiment completed!")