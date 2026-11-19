import re
import json
import hashlib
from pathlib import Path
from datetime import date
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_FOLDER = Path(__file__).parent

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)

all_chunks = []

for file_path in DATA_FOLDER.glob("*.txt"):

    text = file_path.read_text(encoding="utf-8")

    chunks = splitter.split_text(text)

    for index, chunk in enumerate(chunks):

        headings = re.findall(
            r"(?m)^Section \d+:.*$",
            text[:text.find(chunk) + len(chunk)]
        )

        section = headings[-1] if headings else "General"

        metadata = {
            "source": file_path.name,
            "page": 1,
            "section_heading": section,
            "created_date": date.today().isoformat(),
            "chunk_index": index,
            "doc_type": "TXT"
        }

        all_chunks.append({
            "text": chunk,
            "metadata": metadata
        })


# Describe chunk function
def describe_chunk(chunk):

    metadata = chunk["metadata"]

    return (
        f"Source: {metadata['source']} | "
        f"Page: {metadata['page']} | "
        f"Section: {metadata['section_heading']} | "
        f"Chunk: {metadata['chunk_index']}"
    )


print("===== METADATA EXTRACTION =====")

print("Total chunks:", len(all_chunks))

for chunk in all_chunks[:10]:

    print("\nText:", chunk["text"][:100])
    print("Metadata:", json.dumps(
        chunk["metadata"],
        indent=2
    ))

    print("Citation:", describe_chunk(chunk))