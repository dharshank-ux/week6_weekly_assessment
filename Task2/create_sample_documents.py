import fitz
from docx import Document
from pathlib import Path

folder = Path(__file__).parent

# Create PDF document
pdf = fitz.open()
page = pdf.new_page()

page.insert_text(
    (50, 50),
    "COMPANY SAFETY POLICY\n\n"
    "Section 1: Office Safety\n"
    "Employees must follow office safety rules.\n"
    "Emergency exits must remain accessible.\n\n"
    "Section 2: Equipment Safety\n"
    "Employees must handle equipment carefully.\n"
    "Damaged equipment must be reported to IT.",
    fontsize=12
)

pdf.save(folder / "safety_policy.pdf")
pdf.close()

# Create Word document
doc = Document()

doc.add_heading("IT USER MANUAL", 0)

doc.add_heading("Section 1: Computer Usage", level=1)
doc.add_paragraph(
    "Employees must use company computers responsibly."
)

doc.add_heading("Section 2: Technical Support", level=1)
doc.add_paragraph(
    "Employees can contact IT support for technical issues."
)

doc.save(folder / "it_manual.docx")

print("PDF document created successfully.")
print("Word document created successfully.")