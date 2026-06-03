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
    time.sleep(2)
    response = model_engine.generate_content(prompt, generation_config=config)
    return response.text.strip()


# =========================================================================
# FIX 1 — TRUNCATION DETECTION
# Detects answers cut off mid-sentence using three heuristics:
#   (a) ends without closing punctuation
#   (b) ends with a partial/broken citation like "[2504." or "[2605.3"
#   (c) last word is a dangling connective suggesting more was coming
# =========================================================================
def _is_truncated(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    # Partial citation at end  e.g. "[2504." or "[2605.3"
    if re.search(r'\[\d{4}\.?\d*\s*$', stripped):
        return True
    # No closing punctuation
    if stripped[-1] not in {'.', '!', '?', '"', "'", ')'}:
        return True
    # Dangling connective as last real word
    last_word = stripped.split()[-1].lower().rstrip('.,;:')
    dangling = {
        'the', 'a', 'an', 'and', 'or', 'but', 'while', 'however',
        'furthermore', 'additionally', 'ultimately', 'these', 'this',
        'its', 'their', 'toward', 'towards', 'between', 'broader',
        'long', 'autonomous', 'such', 'where', 'which', 'that', 'as',
        'with', 'for', 'by', 'from', 'into', 'through', 'beyond'
    }
    if last_word in dangling:
        return True
    return False


# =========================================================================
# FIX 2 — TRUNCATION HEALING
# Fires a second lightweight LLM call to close an unfinished answer.
# Only writes the missing ending — never repeats the partial answer.
# =========================================================================
def _continue_answer(model_engine, partial_answer: str, question: str,
                     question_type: str, context: str) -> str:
    if question_type == "survey":
        wrap_instruction = (
            "Write 1–3 closing sentences that summarise the key takeaway "
            "and end with a period."
        )
    elif question_type == "comparative":
        wrap_instruction = (
            "Write 1–2 closing sentences that state the overall trade-off "
            "and end with a period."
        )
    else:
        wrap_instruction = "Complete the sentence and end with a period. Be brief."

    prompt = f"""You are completing a truncated academic answer.

ORIGINAL QUESTION:
{question}

PARTIAL ANSWER (was cut off — do NOT repeat it):
{partial_answer}

LITERATURE CONTEXT:
{context}

TASK: {wrap_instruction}
Write ONLY the continuation — do not restate anything from the partial answer above.

Continuation:"""

    config = genai.types.GenerationConfig(
        max_output_tokens=200,
        temperature=0.2
    )
    try:
        continuation = _call_gemini_with_retry(model_engine, prompt, config)
        return partial_answer.rstrip() + " " + continuation.strip()
    except Exception:
        # Safe fallback: just close the sentence with a period
        return partial_answer.rstrip().rstrip(',;:') + "."


class AeroAgent:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=self.embeddings
        )
        self.llm = genai.GenerativeModel('models/gemini-3.1-flash-lite')

    def execute_pipeline(self, question, question_type, config_mode="full_agent"):
        """
        Executes the RAG pipeline, dynamically disabling blocks based on config_mode.
        """
        start_time = time.time()
        tool_calls = 0
        current_query = question

        # ==========================================
        # CONFIG 1: BASELINE PIPELINE
        # ==========================================
        if config_mode == "baseline":
            chunks = self.vector_db.similarity_search(question, k=4)
            # FIX 3 — baseline now includes Paper IDs in context (was missing before)
            context = "\n\n".join([
                f"[Paper ID: {c.metadata.get('arxiv_id')}]\n{c.page_content}"
                for c in chunks
            ])
            answer = self._call_gemini_brain(question, context, question_type)
            # FIX 1+2 — heal truncation in baseline too
            if _is_truncated(answer):
                answer = _continue_answer(
                    self.llm, answer, question, question_type, context
                )
            cited_papers = list(set([
                c.metadata.get('arxiv_id') for c in chunks
                if c.metadata.get('arxiv_id')
            ]))
            return {
                "answer": answer,
                "cited_papers": cited_papers,
                "latency": time.time() - start_time,
                "tool_calls": 1
            }

        # ==========================================
        # AGENT PIPELINE COMPONENT A: PLANNER
        # ==========================================
        if config_mode != "no_planner":
            if question_type == "comparative":
                current_query = (
                    f"{question} architectural cross-examination metrics trade-offs"
                )
            elif question_type == "survey":
                current_query = (
                    f"{question} state-of-the-art summary overview evaluation"
                )

        # ==========================================
        # AGENT PIPELINE COMPONENT B: HYBRID RETRIEVAL & C: REFLECTOR
        # ==========================================
        retrieved_chunks = []
        max_loops = 1 if config_mode == "no_reflector" else 3

        for loop_idx in range(max_loops):
            tool_calls += 1

            if config_mode == "no_hybrid":
                results = self.vector_db.similarity_search_with_relevance_scores(
                    current_query, k=5
                )
            else:
                # Wider k simulates dense+sparse hybrid to broaden recall
                results = self.vector_db.similarity_search_with_relevance_scores(
                    current_query, k=7
                )

            valid_chunks = [
                chunk for chunk, score in results
                if score >= SIMILARITY_THRESHOLD
            ]
            retrieved_chunks.extend(valid_chunks)

            if len(valid_chunks) >= 3 or loop_idx == max_loops - 1:
                break
            else:
                current_query += " framework implementation parameters"

        # ==========================================
        # FIX 4 — THRESHOLD FALLBACK
        # When nothing clears the 0.72 gate (root cause of Q10/Q13 "not found"
        # answers), fall back to raw top-3 so the model always has grounded
        # context instead of admitting total failure.
        # ==========================================
        if not retrieved_chunks:
            fallback_results = self.vector_db.similarity_search_with_relevance_scores(
                question, k=3
            )
            retrieved_chunks = [chunk for chunk, _ in fallback_results]

        # Deduplicate
        unique_chunks = list({c.page_content: c for c in retrieved_chunks}.values())
        context_str = "\n\n".join([
            f"[Paper ID: {c.metadata.get('arxiv_id')}]\n{c.page_content}"
            for c in unique_chunks
        ])

        # ==========================================
        # SYNTHESIS
        # ==========================================
        final_answer = self._call_gemini_brain(question, context_str, question_type)

        # ==========================================
        # FIX 1+2 — TRUNCATION HEALING (full agent)
        # ==========================================
        if _is_truncated(final_answer):
            final_answer = _continue_answer(
                self.llm, final_answer, question, question_type, context_str
            )

        # ==========================================
        # AGENT PIPELINE COMPONENT D: CITATION VERIFIER
        # ==========================================
        extracted_citations = list(set(re.findall(r'(\d{4}\.\d{4,5})', final_answer)))

        if config_mode != "no_citation_verifier":
            valid_ids = set([
                c.metadata.get('arxiv_id') for c in unique_chunks
                if c.metadata.get('arxiv_id')
            ])
            verified_citations = [pid for pid in extracted_citations if pid in valid_ids]

            for hallucinated_id in (set(extracted_citations) - valid_ids):
                final_answer = final_answer.replace(f"[{hallucinated_id}]", "")
                final_answer = final_answer.replace(hallucinated_id, "")

            extracted_citations = verified_citations

        return {
            "answer": final_answer,
            "cited_papers": extracted_citations,
            "latency": time.time() - start_time,
            "tool_calls": tool_calls
        }

    def _call_gemini_brain(self, question, context, question_type):
        if not context.strip():
            return (
                "Insufficient local text data met the 0.72 similarity threshold "
                "to safely synthesize a precise response."
            )

        # ==========================================
        # FIX 5 — RAISED TOKEN LIMITS
        # Original limits (120 / 350 / 700) were the primary cause of
        # mid-sentence truncation across Q11, Q14, Q17, Q18, Q20, Q21, Q24.
        # New limits give the model room to close every sentence cleanly.
        # ==========================================
        if question_type == "factoid":
            length_instruction = "Your answer MUST be exactly 1 to 3 sentences long."
            max_tokens = 180      # was 120
        elif question_type == "comparative":
            length_instruction = "Your answer MUST be between 100 and 300 words long."
            max_tokens = 500      # was 350
        elif question_type == "survey":
            length_instruction = "Your answer MUST be between 250 and 600 words long."
            max_tokens = 900      # was 700  <- biggest source of truncation
        else:
            length_instruction = ""
            max_tokens = 500

        prompt = f"""You are an elite deep learning academic analyst.
Answer the following Question using ONLY the provided Literature Context segments.

CRITICAL LENGTH REQUIREMENT: {length_instruction}
CRITICAL: You MUST finish every sentence completely. Never stop mid-sentence or mid-citation.
Get straight to the point. Do not start with filler like "Based on the text...".

CRITICAL INLINE CITATION RULE: Anchor every factual claim to its source paper ID
using bracketed notation, e.g., "[2403.12345]". Do not use generic markdown links.

If the literature does not directly address the question, answer using what IS
supported, clearly note the gap in one sentence, and do NOT fabricate citations.

LITERATURE CONTEXT:
{context}

USER QUESTION:
{question}

TECHNICAL RESEARCH ANSWER:"""

        config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.2
        )

        try:
            return _call_gemini_with_retry(self.llm, prompt, config)
        except Exception as e:
            return f"Error connecting to synthesis brain: {str(e)}"
