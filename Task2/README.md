Task 2 – Chunking, Metadata and Stable IDs

1. Document Extraction

Created 5 sample TXT documents:

- policy.txt
- employee_manual.txt
- leave_policy.txt
- security_policy.txt
- hr_guidelines.txt

Implemented text extraction using:

- PyMuPDF for PDF
- python-docx for Word
- Plain Python reading for TXT

2. Chunking Strategies

Implemented three chunking methods:

- Fixed-size token chunking with 512 tokens and 50 overlap.
- Sentence-based chunking.
- RecursiveCharacterTextSplitter with chunk_size=512 and chunk_overlap=50.

Compared chunk counts for all three methods.

3. Chunk Size Experiment

Tested four chunk sizes:

- 128
- 256
- 512
- 1024

Used 10 evaluation questions and calculated hit rates.

4. Metadata Extraction

Attached metadata to every chunk:

- Source
- Page
- Section heading
- Created date
- Chunk index
- Document type

Implemented describe_chunk() to generate citation strings.

5. Stable Chunk IDs

Implemented UUID5-based stable chunk IDs.

Tested:

- Run 1: Initial indexing.
- Run 2: Re-indexing without count growth.
- Run 3: Modified one document and checked changed IDs.

6. Technologies Used

- Python
- PyMuPDF
- python-docx
- tiktoken
- LangChain Text Splitters
- Sentence Transformers
- NumPy
- Scikit-learn
- UUID5
- JSON

7. Storage Note

Used a JSON-based simulated vector store for the stable ID and upsert demonstration. Actual vector database storage is covered in Task 3.