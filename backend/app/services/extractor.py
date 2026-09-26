import os
from pypdf import PdfReader
from docx import Document as DocxDocument

class TextExtractionError(Exception):
    """Custom exception raised when text extraction fails."""
    pass

def extract_text_from_file(file_path: str, filename: str) -> str:
    """
    Extract raw text from uploaded files based on file extension (.txt, .pdf, .docx).
    
    Args:
        file_path: Absolute path to the file on local disk.
        filename: Original file name (used to check extension).
        
    Returns:
        Extracted text string.
    """
    extension = os.path.splitext(filename)[1].lower()
    
    try:
        if extension == ".txt":
            return _extract_from_txt(file_path)
        elif extension == ".pdf":
            return _extract_from_pdf(file_path)
        elif extension == ".docx":
            return _extract_from_docx(file_path)
        else:
            raise TextExtractionError(
                f"Unsupported file format '{extension}'. Supported formats: .txt, .pdf, .docx"
            )
    except Exception as e:
        if isinstance(e, TextExtractionError):
            raise e
        raise TextExtractionError(f"Failed to extract text from '{filename}': {str(e)}")

def _extract_from_txt(file_path: str) -> str:
    """Read plain text files with UTF-8 encoding (with fallback)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback encoding if UTF-8 fails
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()

def _extract_from_pdf(file_path: str) -> str:
    """Extract text from PDF pages using PyPDF."""
    reader = PdfReader(file_path)
    extracted_pages = []
    
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            extracted_pages.append(text)
            
    full_text = "\n\n".join(extracted_pages).strip()
    if not full_text:
        raise TextExtractionError("No readable text found in PDF file. (File might contain scanned images without OCR).")
    return full_text

def _extract_from_docx(file_path: str) -> str:
    """Extract text from Word DOCX paragraphs."""
    doc = DocxDocument(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs).strip()
    
    if not full_text:
        raise TextExtractionError("No readable text found in DOCX file.")
    return full_text
