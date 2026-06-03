import os

# System Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PDF_DIR = os.path.join(BASE_DIR, "data", "raw_pdfs")
PARSED_TEXT_DIR = os.path.join(BASE_DIR, "data", "parsed_text")
DB_DIR = os.path.join(BASE_DIR, "db")
EVAL_FILE = os.path.join(BASE_DIR, "eval", "questions.jsonl")

# Optimization Thresholds for Maximum Accuracy
SIMILARITY_THRESHOLD = 0.72   
VALIDATION_THRESHOLD = 0.80   
MAX_REFLECTION_LOOPS = 3