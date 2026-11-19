import tiktoken

from pathlib import Path

# Load documents
DATA_FOLDER = Path(__file__).parent

documents = {}

for file_path in DATA_FOLDER.glob("*.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        documents[file_path.name] = file.read()


# Tokenizer
encoding = tiktoken.get_encoding("cl100k_base")


# Fixed-size token chunking
def fixed_size_chunk(text, chunk_size=512, overlap=50):

    tokens = encoding.encode(text)

    chunks = []

    start = 0

    while start < len(tokens):

        end = start + chunk_size

        chunk_tokens = tokens[start:end]

        chunks.append(
            encoding.decode(chunk_tokens)
        )

        start += chunk_size - overlap

    return chunks


# Test fixed-size chunking
print("===== FIXED SIZE TOKEN CHUNKING =====")

for filename, content in documents.items():

    chunks = fixed_size_chunk(
        content,
        chunk_size=512,
        overlap=50
    )

    print("\nDocument:", filename)
    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:", chunk[:100])
        
        import re

# Sentence-based chunking
def sentence_chunk(text, max_sentences=3):

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []

    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        chunks.append(chunk)

    return chunks


print("\n===== SENTENCE BASED CHUNKING =====")

for filename, content in documents.items():

    chunks = sentence_chunk(content, max_sentences=3)

    print("\nDocument:", filename)
    print("Total sentence chunks:", len(chunks))

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:", chunk[:150])
        
        # RECURSIVE CHARACTER CHUNKING

from langchain_text_splitters import RecursiveCharacterTextSplitter

print("\n===== RECURSIVE CHARACTER CHUNKING =====")

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

for filename, content in documents.items():

    chunks = recursive_splitter.split_text(content)

    print("\nDocument:", filename)
    print("Total recursive chunks:", len(chunks))

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:", chunk[:150])
        
        # CHUNKING STRATEGY COMPARISON

print("\n===== CHUNKING COMPARISON =====")

total_fixed = 0
total_sentence = 0
total_recursive = 0

for filename, content in documents.items():

    fixed_chunks = fixed_size_chunk(
        content, chunk_size=512, overlap=50
    )

    sentence_chunks = sentence_chunk(
        content, max_sentences=3
    )

    recursive_chunks = recursive_splitter.split_text(content)

    total_fixed += len(fixed_chunks)
    total_sentence += len(sentence_chunks)
    total_recursive += len(recursive_chunks)

print("Fixed-size token chunks:", total_fixed)
print("Sentence-based chunks:", total_sentence)
print("Recursive character chunks:", total_recursive)


# SHOW A SENTENCE CUT AT A CHUNK BOUNDARY

print("\n===== CHUNK BOUNDARY DEMONSTRATION =====")

sample_text = (
    "Employees must complete their work on time. "
    "Managers should review employee performance regularly. "
    "All employees must follow company rules."
)

sample_chunks = fixed_size_chunk(
    sample_text,
    chunk_size=8,
    overlap=0
)

for i, chunk in enumerate(sample_chunks):
    print(f"Chunk {i+1}:", chunk)