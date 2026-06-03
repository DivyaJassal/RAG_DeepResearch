# src/agent.py
import re
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
import google.api_core.exceptions
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import DB_DIR, SIMILARITY_THRESHOLD
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure Gemini with your API Key

load_dotenv()

genai.configure(
    api_key=os.getenv("GENAI_API_KEY")
)

# =========================================================================
# QUOTA SAFETY LAYER: Handles Rate-Limits & 429 Errors Automatically
# =========================================================================
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted),
    reraise=True
)
def _call_gemini_with_retry(model_engine, prompt, config):
    """
    Safely executes the prompt, introducing a 2-second pacing window 
    and catching quota errors to automatically backoff and retry.
    """
    time.sleep(2)  # Enforces a baseline spacing to respect free RPM limits
    response = model_engine.generate_content(prompt, generation_config=config)
    return response.text.strip()


class AeroAgent:
    def __init__(self):
        # Native semantic retrieval model (Keeping your essential local chunks database active)
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={'device': 'cpu'})
        self.vector_db = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
        
        # Using gemini-3.1-flash-lite as its free tier supports high-volume daily queries
        self.llm = genai.GenerativeModel('models/gemini-3.1-flash-lite')
        
    def execute_pipeline(self, question, question_type, config_mode="full_agent"):
        """
        Executes the RAG pipeline dynamically disabling blocks based on config_mode.
        """
        start_time = time.time()
        tool_calls = 0
        current_query = question

        # ==========================================
        # CONFIG 1: BASELINE PIPELINE
        # ==========================================
        if config_mode == "baseline":
            # Pure single-shot vector retrieval + 1 LLM call
            chunks = self.vector_db.similarity_search(question, k=4)
            context = "\n\n".join([c.page_content for c in chunks])
            answer = self._call_gemini_brain(question, context, question_type)
            cited_papers = list(set([c.metadata.get('arxiv_id') for c in chunks if 'arxiv_id' in c.metadata]))
            return {"answer": answer, "cited_papers": cited_papers, "latency": time.time() - start_time, "tool_calls": 1}

        # ==========================================
        # AGENT PIPELINE COMPONENT A: PLANNER
        # ==========================================
        if config_mode != "no_planner":
            # Simple keyword optimizer step mimicking query decomposition
            if question_type == "comparative":
                current_query = f"{question} architectural cross-examination metrics trade-offs"
            elif question_type == "survey":
                current_query = f"{question} state-of-the-art summary overview evaluation"

        # ==========================================
        # AGENT PIPELINE COMPONENT B: HYBRID RETRIEVAL & C: REFLECTOR
        # ==========================================
        retrieved_chunks = []
        # If no reflector, limit strictly to 1 search loop iteration
        max_loops = 1 if config_mode == "no_reflector" else 3
        
        for loop_idx in range(max_loops):
            tool_calls += 1
            
            if config_mode == "no_hybrid":
                # Standard pure vector search lookup
                results = self.vector_db.similarity_search_with_relevance_scores(current_query, k=5)
            else:
                # Simulated Hybrid Retrieval (Simulates Dense + Sparse keywords using wider bounds)
                results = self.vector_db.similarity_search_with_relevance_scores(current_query, k=7)

            # Keep only chunks passing your 0.72 certainty gate threshold
            valid_chunks = [chunk for chunk, score in results if score >= SIMILARITY_THRESHOLD]
            retrieved_chunks.extend(valid_chunks)
            
            # Reflector check: If we have enough valid chunks or ran out of attempts, break
            if len(valid_chunks) >= 3 or loop_idx == max_loops - 1:
                break
            else:
                # Re-planning keyword shift for next loop
                current_query += " framework implementation parameters"

        # Deduplicate retrieved segments
        unique_chunks = list({c.page_content: c for c in retrieved_chunks}.values())
        context_str = "\n\n".join([f"[Paper ID: {c.metadata.get('arxiv_id')}]\n{c.page_content}" for c in unique_chunks])

        # Synthesize final text response using Gemini
        final_answer = self._call_gemini_brain(question, context_str, question_type)

        # ==========================================
        # AGENT PIPELINE COMPONENT D: CITATION VERIFIER
        # ==========================================
        extracted_citations = list(set(re.findall(r'(\d{4}\.\d{4,5})', final_answer)))
        
        if config_mode != "no_citation_verifier":
            # Validate citations against chunks that were physically retrieved
            valid_ids = set([c.metadata.get('arxiv_id') for c in unique_chunks if 'arxiv_id' in c.metadata])
            verified_citations = [pid for pid in extracted_citations if pid in valid_ids]
            
            # Cleanly scrub any hallucinated citation tags from the text
            for hallucinated_id in (set(extracted_citations) - valid_ids):
                final_answer = final_answer.replace(f"[{hallucinated_id}]", "").replace(hallucinated_id, "")
            
            extracted_citations = verified_citations

        return {
            "answer": final_answer,
            "cited_papers": extracted_citations,
            "latency": time.time() - start_time,
            "tool_calls": tool_calls
        }

    def _call_gemini_brain(self, question, context, question_type):
        if not context.strip():
            return "Insufficient local text data met the 0.72 similarity threshold to safely synthesize a precise response."
            
        # Dynamically append length guidelines directly into the prompt instructions
        if question_type == "factoid":
            length_instruction = "Your answer MUST be exceptionally brief: exactly 1 to 3 sentences long."
        elif question_type == "comparative":
            length_instruction = "Your answer MUST be balanced: exactly between 100 and 300 words long."
        elif question_type == "survey":
            length_instruction = "Your answer MUST be highly comprehensive: exactly between 250 and 600 words long."
        else:
            length_instruction = ""

        prompt = f"""You are an elite deep learning academic analyst. 
Answer the following Question utilizing ONLY the provided Literature Context segments.

CRITICAL LENGTH REQUIREMENT: {length_instruction}
Get straight to the point. Do not start with conversational filler like 'Based on the text...'.

CRITICAL INLINE CITATION RULE: You must anchor every claim explicitly to its source paper ID using bracketed notation, e.g., "[2403.12345]". Do not use generic markdown links.

LITERATURE CONTEXT:
{context}

USER QUESTION:
{question}

TECHNICAL RESEARCH ANSWER:"""
        
        # Configure token limits dynamically as a secondary length safety rail
        config = genai.types.GenerationConfig(
            max_output_tokens=700 if question_type == "survey" else (350 if question_type == "comparative" else 120),
            temperature=0.2  # Low temperature forces truthfulness over rambling
        )

        try:
            return _call_gemini_with_retry(self.llm, prompt, config)
        except Exception as e:
            return f"Error connecting to synthesis brain: {str(e)}"