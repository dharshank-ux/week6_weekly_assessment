import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_similarity

# 10 sample sentences
sentences = [
    "The cat is sleeping on the sofa.",
    "A kitten is resting on the couch.",
    "The dog is playing in the garden.",
    "Python is a popular programming language.",
    "Machine learning helps computers learn.",
    "Artificial intelligence is transforming technology.",
    "The weather is sunny today.",
    "It is a bright and warm day.",
    "I enjoy reading books.",
    "The car is parked outside the house."
]

# Load local embedding model
print("Loading local embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
start = time.time()
embeddings = model.encode(sentences)
local_time = time.time() - start

print("\nLOCAL EMBEDDINGS")
print("Number of sentences:", len(sentences))
print("Embedding dimensions:", embeddings.shape)
print("Time taken:", round(local_time, 4), "seconds")


# Custom cosine similarity function
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# Test cosine similarity
print("\nCOSINE SIMILARITY TESTS")

tests = [
    ([1, 2, 3], [2, 4, 6]),
    ([1, 0], [0, 1]),
    ([1, 0], [-1, 0])
]

for a, b in tests:
    print(a, "vs", b, "->", round(cosine_similarity(a, b), 2))


# Compare with Scikit-learn
print("\nSCIKIT-LEARN COMPARISON")

for a, b in tests:
    result = sklearn_similarity([a], [b])[0][0]
    print("Similarity:", round(float(result), 2))


# Similarity matrix
print("\nSIMILARITY MATRIX")

similarity_matrix = sklearn_similarity(embeddings)

print(np.round(similarity_matrix, 2))


# Find top 3 similar pairs
print("\nTOP 3 SIMILAR PAIRS")

pairs = []

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        pairs.append((similarity_matrix[i][j], i, j))

pairs.sort(reverse=True)

for score, i, j in pairs[:3]:
    print(f"{sentences[i]} <-> {sentences[j]}")
    print("Similarity:", round(float(score), 3))


# No common words but high semantic similarity
print("\nSEMANTIC SIMILARITY EXAMPLE")

s1 = "The automobile is very fast."
s2 = "The vehicle moves quickly."

e1, e2 = model.encode([s1, s2])

print("Sentence 1:", s1)
print("Sentence 2:", s2)
print("Similarity:", round(cosine_similarity(e1, e2), 3))

# USE CASE 1: SEMANTIC SEARCH

print("\n===== SEMANTIC SEARCH =====")

queries = [
    "Animals resting",
    "Programming and technology",
    "Sunny weather"
]

for query in queries:
    query_embedding = model.encode([query])
    scores = sklearn_similarity(query_embedding, embeddings)[0]

    best_indices = np.argsort(scores)[::-1][:2]

    print("\nQuery:", query)

    for index in best_indices:
        print(
            sentences[index],
            "-> Score:",
            round(float(scores[index]), 3)
        )


# USE CASE 2: DUPLICATE DETECTION

print("\n===== DUPLICATE DETECTION =====")

threshold = 0.75

for score, i, j in pairs:
    if score >= threshold:
        print("Possible duplicate:")
        print(sentences[i])
        print(sentences[j])
        print("Similarity:", round(float(score), 3))


# USE CASE 3: K-MEANS CLUSTERING

print("\n===== K-MEANS CLUSTERING =====")

from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(embeddings)

cluster_names = {
    0: "Technology",
    1: "Daily Life",
    2: "Animals and Nature"
}

for i, sentence in enumerate(sentences):
    print(
        cluster_names[labels[i]],
        "->",
        sentence
    )


# LOCAL VS API MODEL DECISION

print("\n===== MODEL SELECTION =====")

print("1. Local models can run without API credentials.")
print("2. API models require an API key and internet access.")
print("3. Local models offer better control over data privacy.")
print("4. API models can provide managed infrastructure and scalability.")

# LOCAL MODEL VS API MODEL COMPARISON

import os

print("\n===== LOCAL VS API COMPARISON =====")

print("Local Model: all-MiniLM-L6-v2")
print("Local Dimensions:", embeddings.shape[1])
print("Local Time:", round(local_time, 4), "seconds")

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    start = time.time()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=sentences
    )

    api_time = time.time() - start

    api_embeddings = np.array(
        [item.embedding for item in response.data]
    )

    print("API Model: text-embedding-3-small")
    print("API Dimensions:", api_embeddings.shape)
    print("API Time:", round(api_time, 4), "seconds")

else:
    print("API key not available.")
    print("API comparison skipped.")
    print("Local embedding completed successfully.")