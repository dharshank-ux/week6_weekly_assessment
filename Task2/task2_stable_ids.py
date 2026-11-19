import hashlib
import json
import uuid

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_FOLDER = Path(__file__).parent

# Stable namespace
NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)


def create_chunks():

    all_chunks = []

    for file_path in sorted(DATA_FOLDER.glob("*.txt")):

        text = file_path.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)

        for index, chunk in enumerate(chunks):

            chunk_id = str(
                uuid.uuid5(
                    NAMESPACE,
                    f"{file_path.name}:{index}:{chunk}"
                )
            )

            all_chunks.append({
                "id": chunk_id,
                "source": file_path.name,
                "chunk_index": index,
                "text": chunk
            })

    return all_chunks


# Simulated vector database using JSON
def save_chunks(chunks):

    database = {
        item["id"]: item
        for item in chunks
    }

    with open(
        DATA_FOLDER / "vector_store.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(database, file, indent=2)


def load_chunks():

    path = DATA_FOLDER / "vector_store.json"

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


print("===== STABLE ID TEST =====")

# Run 1
chunks_run1 = create_chunks()
save_chunks(chunks_run1)

database1 = load_chunks()

print("Run 1:", len(database1), "points")

# Run 2
chunks_run2 = create_chunks()
save_chunks(chunks_run2)

database2 = load_chunks()

print("Run 2:", len(database2), "points")

print(
    "Count unchanged:",
    len(database1) == len(database2)
)

print(
    "IDs unchanged:",
    set(database1.keys()) == set(database2.keys())
)

# Run 3 - Modify one paragraph
test_file = DATA_FOLDER / "policy.txt"

original_text = test_file.read_text(encoding="utf-8")

modified_text = original_text + (
    "\nSection 4: Updated Policy\n"
    "Employees must follow updated company guidelines."
)

test_file.write_text(modified_text, encoding="utf-8")

chunks_run3 = create_chunks()

old_ids = set(database2.keys())
new_ids = {item["id"] for item in chunks_run3}

print("\nRun 3 after one edit")

print("Run 3:", len(new_ids), "points")
print("New or changed IDs:", len(new_ids - old_ids))
print("Removed IDs:", len(old_ids - new_ids))

# Restore original document
test_file.write_text(original_text, encoding="utf-8")

print("\nOriginal document restored.")