# Configuration Comparison

## Objective
Compare semantic search performance using two chunk-size configurations.

## Configuration 1

- Chunk Size: 256
- Chunk Overlap: 50
- Embedding Model: all-MiniLM-L6-v2

## Configuration 2

- Chunk Size: 512
- Chunk Overlap: 50
- Embedding Model: all-MiniLM-L6-v2

## Evaluation Metrics

| Metric | Chunk Size 256 | Chunk Size 512 |
|---|---|---|
| Hit@1 | Record actual result | Record actual result |
| Hit@3 | Record actual result | Record actual result |
| Hit@5 | Record actual result | Record actual result |
| MRR | Record actual result | Record actual result |

## Observations

- Both configurations use the same embedding model.
- Both configurations use 50-token overlap.
- Chunk size can affect retrieval results.
- Actual evaluation metrics are used for comparison.

## Conclusion

The configuration with better evaluation metrics can be selected for semantic document search.