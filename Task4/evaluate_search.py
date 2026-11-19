import json
import re
import urllib.request
from pathlib import Path

API_URL = "http://127.0.0.1:8000/search"
API_KEY = "week6-local-secret-key"

DOCS_DIR = Path(__file__).resolve().parent.parent / "Task2"

def create_evaluation_queries():
    queries = []

    for file_path in sorted(DOCS_DIR.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8")
        sentences = re.split(r"(?<=[.!?])\s+", content)

        valid_sentences = [
            s.strip() for s in sentences
            if len(s.strip().split()) >= 5
            and not s.strip().startswith("Section")
        ]

        for sentence in valid_sentences[:3]:
            words = sentence.split()
            query = " ".join(words[:min(8, len(words))])

            queries.append({
                "query": query,
                "expected_source": file_path.name
            })

    return queries[:15]


def search(query):
    body = json.dumps({
        "query": query,
        "top_k": 5
    }).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": API_KEY
        },
        method="POST"
    )

    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    evaluation_queries = create_evaluation_queries()

    if len(evaluation_queries) < 15:
        print("Warning: Less than 15 queries generated.")

    hits = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []

    print("\nSEARCH EVALUATION RESULTS\n")

    for index, item in enumerate(evaluation_queries, 1):
        response = search(item["query"])
        results = response.get("results", [])

        sources = [
            Path(result.get("source", "")).name
            for result in results
        ]

        expected = item["expected_source"]

        rank = next(
            (i for i, source in enumerate(sources, 1)
             if source == expected),
            None
        )

        if rank:
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

        for k in hits:
            if expected in sources[:k]:
                hits[k] += 1

        print(f"\nQuery {index}: {item['query']}")
        print(f"Expected source: {expected}")
        print(f"Expected page: TXT page metadata is synthetic")
        print(f"Retrieved sources: {sources}")
        print(f"Correct source rank: {rank}")

    total = len(evaluation_queries)

    print("\n========== FINAL METRICS ==========")

    for k in [1, 3, 5]:
        score = hits[k] / total if total else 0
        print(f"Hit@{k}: {score:.2%}")

    mrr = sum(reciprocal_ranks) / total if total else 0
    print(f"MRR: {mrr:.4f}")

    print(f"Total queries: {total}")


if __name__ == "__main__":
    main()