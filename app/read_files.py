from PyPDF2 import PdfReader
from docx import Document

def sanitize_text(text: str) -> str:
    return text.replace('\x00', '')

def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return sanitize_text(text)

def read_docx(file_path):
    doc = Document(file_path)
    return  sanitize_text("\n".join([para.text for para in doc.paragraphs]))