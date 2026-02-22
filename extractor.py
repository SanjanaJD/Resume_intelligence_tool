import io
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from docx import Document


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract text from PDF or DOCX files, with OCR fallback for scanned PDFs."""
    if filename.lower().endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        # Fallback to OCR for scanned PDFs
        if len(text.strip()) < 100:
            try:
                images = convert_from_bytes(file_bytes)
                text = "\n".join(pytesseract.image_to_string(img) for img in images)
            except Exception:
                pass  # If OCR fails, return whatever pdfplumber got
        return text
    elif filename.lower().endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    return ""