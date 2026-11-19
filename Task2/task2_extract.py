import fitz
from docx import Document
from pathlib import Path

DATA_FOLDER = Path(__file__).parent


def extract_txt(file_path):
    return file_path.read_text(encoding="utf-8")


def extract_pdf(file_path):
    text = ""

    with fitz.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text += f"\nPage {page_number}\n"
            text += page.get_text()

    return text


def extract_docx(file_path):
    document = Document(file_path)

    return "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
    )


def extract_document(file_path):

    extension = file_path.suffix.lower()

    if extension == ".txt":
        return extract_txt(file_path)

    elif extension == ".pdf":
        return extract_pdf(file_path)

    elif extension == ".docx":
        return extract_docx(file_path)

    return None


print("===== ALL DOCUMENT EXTRACTION =====")

supported = [".txt", ".pdf", ".docx"]

for file_path in sorted(DATA_FOLDER.iterdir()):

    if file_path.suffix.lower() in supported:

        content = extract_document(file_path)

        print("\nFile:", file_path.name)
        print("Characters:", len(content))
        print("Preview:", content[:150])
        print("Extraction successful:", bool(content.strip()))

print("\nDOCUMENT EXTRACTION COMPLETED")