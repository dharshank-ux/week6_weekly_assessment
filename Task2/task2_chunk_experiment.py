import numpy as np
import tiktoken
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load documents
folder = Path(__file__).parent
documents = {}

for file in folder.glob("*.txt"):
    documents[file.name] = file.read_text(encoding="utf-8")

# 10 evaluation questions with expected source
questions = [
    ("How many hours should employees work?", "policy.txt"),
    ("What are the office working hours?", "policy.txt"),
    ("How should employees communicate?", "employee_manual.txt"),
    ("When is employee performance reviewed?", "employee_manual.txt"),
    ("How many annual leave days are allowed?", "leave_policy.txt"),
    ("Who approves annual leave?", "leave_policy.txt"),
    ("How should passwords be protected?", "security_policy.txt"),
    ("What should employees do with suspicious emails?", "security_policy.txt"),
    ("What happens during employee onboarding?", "hr_guidelines.txt"),
    ("Who verifies candidate qualifications?", "hr_guidelines.txt")
]

encoding = tiktoken.get_encoding("cl100k_base")

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_chunks(text, size, overlap=50):
    tokens = encoding.encode(text)
    chunks = []

    step = max(1, size - overlap)

    for start in range(0, len(tokens), step):
        chunk = encoding.decode(tokens[start:start + size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


print("===== CHUNK SIZE EXPERIMENT =====")

for size in [128, 256, 512, 1024]:

    chunk_data = []

    for source, text in documents.items():
        chunks = create_chunks(text, size)

        for chunk in chunks:
            chunk_data.append({
                "source": source,
                "page": 1,
                "text": chunk
            })

    chunk_embeddings = model.encode(
        [item["text"] for item in chunk_data]
    )

    hits = 0

    for question, expected_source in questions:

        query_embedding = model.encode([question])

        scores = cosine_similarity(
            query_embedding,
            chunk_embeddings
        )[0]

        best_index = np.argmax(scores)
        result = chunk_data[best_index]

        if result["source"] == expected_source:
            hits += 1

    hit_rate = hits / len(questions)

    print(
        f"Chunk Size: {size} | "
        f"Hits: {hits}/10 | "
        f"Hit Rate: {hit_rate:.2f}"
    )