from pathlib import Path
import pymupdf
from pymupdf import Document

PDF_PATH = Path("data", "test")
doc = pymupdf.open(Path(PDF_PATH, "intel_test.pdf"))
out = open("output.txt", "wb") # create a text output
for page in doc: # iterate the document pages
    text = page.get_text().encode("utf8") # get plain text (is in UTF-8)
    out.write(text) # write text of page
    out.write(bytes((12,))) # write page delimiter (form feed 0x0C)
out.close()

# Load page
page = doc.load_page(1)
text = page.get_text()

# Generator function for pdf text

def get_pdf_text(fname: str | Path) -> str:
    """This function takes the filename of a PDF and yields the 
    content of the PDF as str

    Args:
        fname (str | Path): Path to PDF file as str or pathlib Path

    Returns:
        str: Text of pdf
    """

    doc = pymupdf.open(fname)
    pages = doc.page_count
    for page in pages:
        text = page.get_text().encode("utf8")       # Check best encoding for RAG

def _get_page(doc: Document) -> str:
    """Generator function that returns text of single page
    from PyMuPDF document

    Args:
        doc (Document): PyMuPDF document object

    Yields:
        str: Text of page as string
    """

    pages = doc.page_count
    for page in pages:
        yield page.get_text().encode("utf8")       # Check best encoding for RAG

