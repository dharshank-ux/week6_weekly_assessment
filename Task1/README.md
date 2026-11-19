Task 1 – Embeddings and Cosine Similarity

Objective

To understand embeddings, cosine similarity, semantic search, duplicate detection and clustering.

1. Embedding Model

- Model: all-MiniLM-L6-v2
- Number of sentences: 10
- Embedding dimensions: 384
- Library: Sentence Transformers

2. Cosine Similarity

Implemented cosine similarity from scratch using NumPy.

Test Results:

- [1,2,3] vs [2,4,6] = 1.0
- [1,0] vs [0,1] = 0.0
- [1,0] vs [-1,0] = -1.0

Compared results with Scikit-learn.

3. Similarity Matrix

Generated a similarity matrix for all 10 sentences and identified the top 3 most similar pairs.

4. Semantic Similarity

Tested two sentences with different words but similar meanings.

Result: 0.831

5. Use Cases

- Semantic Search – Tested 3 queries.
- Duplicate Detection – Used a similarity threshold of 0.75.
- Clustering – Used K-Means with 3 named clusters.

6. Local vs API Models

- Local models can run without API credentials.
- API models require an API key and internet access.
- Local models provide better control over data privacy.
- API models provide managed infrastructure and scalability.

7. API Comparison

Local embedding completed successfully using all-MiniLM-L6-v2.

OpenAI API comparison code is included. Actual API execution is pending because an API key is not available.

Technologies Used

- Python
- NumPy
- Sentence Transformers
- Scikit-learn
- OpenAI SDK