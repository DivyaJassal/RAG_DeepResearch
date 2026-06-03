# src/db_indexer.py
import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import PARSED_TEXT_DIR, DB_DIR

def build_vector_database():
    print("--- INITIATING VECTOR INDEX GENERATION MATRIX ---")
    
    print("Loading HuggingFace Embedding pipeline (BAAI/bge-small-en-v1.5)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    
    documents = []
    metadatas = []
    
    if not os.path.exists(PARSED_TEXT_DIR):
        print(f"Error: {PARSED_TEXT_DIR} does not exist yet. Wait for the indexer to finish!")
        return
        
    text_files = [f for f in os.listdir(PARSED_TEXT_DIR) if f.endswith(".txt")]
    print(f"Found {len(text_files)} text papers ready for vector injection.")
    
    for filename in text_files:
        arxiv_id = filename.replace(".txt", "")
        file_path = os.path.join(PARSED_TEXT_DIR, filename)
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        if not content.strip():
            continue
            
        # Hierarchical Chunking strategy
        chunks = [content[i:i+600] for i in range(0, len(content), 400)]
        
        for chunk in chunks:
            documents.append(chunk)
            metadatas.append({"arxiv_id": arxiv_id})
            
    print(f"Total text segments generated: {len(documents)}. Commencing database embedding compilation...")
    
    vector_db = Chroma.from_texts(
        texts=documents,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=DB_DIR
    )
    
    print(f"--- SUCCESS: Vector store safely persisted locally at {DB_DIR} ---")

if __name__ == "__main__":
    build_vector_database()