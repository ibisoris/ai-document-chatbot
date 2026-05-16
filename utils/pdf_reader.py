from pathlib import Path

from pypdf import PdfReader

from utils.text_chunker import clean_extracted_text


class PdfExtractionError(Exception):
    """Raised when text cannot be extracted from a PDF."""


def get_pdf_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception as error:
        raise PdfExtractionError("The PDF could not be opened or read.") from error


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract readable text from each page in a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        pages_text = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)

        return clean_extracted_text("\n\n".join(pages_text))
    except Exception as error:
        raise PdfExtractionError("The PDF text could not be extracted.") from error
