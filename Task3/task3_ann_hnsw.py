import faiss
import numpy as np
import time

# Generate 50,000 random vectors
np.random.seed(42)

num_vectors = 50000
dimension = 384
num_queries = 100
k = 10

vectors = np.random.random(
    (num_vectors, dimension)
).astype("float32")

queries = np.random.random(
    (num_queries, dimension)
).astype("float32")

# Normalize vectors
faiss.normalize_L2(vectors)
faiss.normalize_L2(queries)

# 1. Exact Search using Flat Index
exact_index = faiss.IndexFlatIP(dimension)

start = time.time()
exact_index.add(vectors)
exact_build_time = time.time() - start

start = time.time()
exact_distances, exact_results = exact_index.search(queries, k)
exact_search_time = time.time() - start

# 2. Approximate Search using HNSW
hnsw_index = faiss.IndexHNSWFlat(
    dimension, 32, faiss.METRIC_INNER_PRODUCT
)

hnsw_index.hnsw.efConstruction = 100
hnsw_index.hnsw.efSearch = 64

start = time.time()
hnsw_index.add(vectors)
hnsw_build_time = time.time() - start

start = time.time()
hnsw_distances, hnsw_results = hnsw_index.search(queries, k)
hnsw_search_time = time.time() - start

# 3. Calculate Recall@10
recall_values = []

for exact, approximate in zip(exact_results, hnsw_results):
    recall = len(set(exact) & set(approximate)) / k
    recall_values.append(recall)

recall_at_10 = np.mean(recall_values)

# 4. Print Results
print("\nANN and HNSW Experiment")
print("-----------------------")
print("Total vectors:", num_vectors)
print("Vector dimension:", dimension)
print("Queries:", num_queries)

print("\nExact Search:")
print("Build time:", round(exact_build_time, 4), "seconds")
print("Search time:", round(exact_search_time, 4), "seconds")

print("\nHNSW Search:")
print("Build time:", round(hnsw_build_time, 4), "seconds")
print("Search time:", round(hnsw_search_time, 4), "seconds")

print("\nRecall@10:", round(recall_at_10, 4))

print("\nExperiment completed successfully!")