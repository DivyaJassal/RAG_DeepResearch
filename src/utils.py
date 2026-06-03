import fitz  # PyMuPDF
import re

def extract_academic_text(pdf_path):
    """
    Sequentially reads structural boxes downward by columns to bypass 
    multi-column parsing errors, and crops files when bibliographies begin.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"File system skipped structural read failure on {pdf_path}: {e}")
        return ""
        
    clean_text_blocks = []
    for page in doc:
        # sort=True tells PyMuPDF to read downward in column 1 first, then column 2
        blocks = page.get_text("blocks", sort=True)
        for b in blocks:
            text_fragment = b[4].strip()
            
            # Look for reference and bibliography headers to cut off text extraction
            normalized = text_fragment.lower()
            if "## references" in normalized or "## bibliography" in normalized or normalized == "references":
                doc.close()
                return "\n".join(clean_text_blocks)
                
            clean_text_blocks.append(text_fragment)
            
    doc.close()
    return "\n".join(clean_text_blocks)