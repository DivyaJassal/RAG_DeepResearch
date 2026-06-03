import os
from src.config import RAW_PDF_DIR, PARSED_TEXT_DIR
from src.corpus_builder import build_and_download_corpus
from src.utils import extract_academic_text

def run_pipeline():
    # Phase 1: Build directories and download 800 documents
    print("--- STARTING ACADEMIC SCHOLASTIC COLLECTION LOOP ---")
    build_and_download_corpus(target_count=800)
    
    # Phase 2: Structural Column Alignment Conversion
    print("\n--- INITIATING TEXT STRUCTURAL CONVERSION MATRIX ---")
    os.makedirs(PARSED_TEXT_DIR, exist_ok=True)
    
    all_pdfs = [f for f in os.listdir(RAW_PDF_DIR) if f.endswith(".pdf")]
    print(f"Discovered {len(all_pdfs)} raw target nodes cached inside directory.")
    
    for idx, filename in enumerate(all_pdfs):
        arxiv_id = filename.replace(".pdf", "")
        output_txt_path = os.path.join(PARSED_TEXT_DIR, f"{arxiv_id}.txt")
        
        # Skip processing if text file already exists
        if os.path.exists(output_txt_path):
            continue
            
        pdf_full_path = os.path.join(RAW_PDF_DIR, filename)
        print(f"[{idx+1}/{len(all_pdfs)}] Converting document tracking matrix: {filename}")
        
        parsed_content = extract_academic_text(pdf_full_path)
        
        with open(output_txt_path, "w", encoding="utf-8") as text_file:
            text_file.write(parsed_content)
            
    print("\n--- CORPUS PIPELINE OPERATION COMPLETE: 800 PAPERS PARSED ---")

if __name__ == "__main__":
    run_pipeline()