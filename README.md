# AeroAgent: An Autonomous RAG System for Deep Learning Literature Analysis

AeroAgent is an advanced, autonomous Retrieval-Augmented Generation (RAG) agent engineered to systematically parse, cross-examine, and synthesize insights from dense deep learning academic literature. By transforming traditional, rigid linear RAG lookups into an adaptive multi-component agent framework, AeroAgent dynamically self-corrects search queries, enforces strict textual length constraints, and validates citations against physical data tracks.

---

## 🚀 Key Architectural Highlights

* **Adaptive 4-Stage Agent Pipeline:** Orchestrates query planning, hybrid semantic retrieval, reflective validation, and citation cross-verification.
* **Production-Grade Resilience:** Embedded with automated exponential backoff mechanisms utilizing `tenacity` to seamlessly handle `429 ResourceExhausted` API quota boundaries.
* **Dynamic Response Structuring:** Built-in programmatic length controllers ensuring tight alignment with rigorous academic submission guidelines (Factoid, Comparative, and Survey formats).
* **Robust Citation Guard:** Active regex parsing engine that sanitizes output payloads by scrubbing hallucinated citations that do not exist within the physically retrieved context chunks.

---

## 📊 System Performance Curve

Through rigorous multi-scenario ablation testing across 30 complex analytical queries, the agent demonstrates a highly optimized efficiency-to-accuracy balance:

| Metric | System Profile | Operational Significance |
| :--- | :--- | :--- |
| **Average Latency** | **3.958 seconds** | Balanced pacing window ensuring zero API drops while capturing deep context. |
| **Synthesis Backbone** | `gemini-3.1-flash-lite` | High-volume free tier allowance supporting extended multi-loop executions. |
| **Retrieval Threshold** | $\ge$ **0.72 Similarity** | Rigid certainty gate keeping the generation strictly grounded. |

---

## 🛠️ Project Architecture

```text
.
├── src/
│   ├── agent.py          # Core AeroAgent engine & resilience layers
│   ├── evaluator.py      # Automated multi-mode ablation sweep harness
│   └── config.py         # Global variables, DB paths, and threshold gates
├── predictions/          # Generated JSONL data traces per ablation variant
│   ├── baseline.jsonl
│   ├── no_planner.jsonl
│   ├── no_hybrid.jsonl
│   ├── no_reflector.jsonl
│   ├── no_citation_verifier.jsonl
│   └── full_agent.jsonl
└── README.md

🔬 Ablation Framework Matrix
AeroAgent evaluates the performance contributions of its system modules by systematically executing the pipeline under 6 specialized ablation configurations:

baseline: Pure single-shot vector retrieval combined with 1 direct LLM call.

no_planner: Disables contextual query decomposition and keyword optimization.

no_hybrid: Limits retrieval strictly to native vector searches, removing keyword expansion.

no_reflector: Forces the agent to settle for initial data reads, capping the search loop at 1 cycle.

no_citation_verifier: Disables post-generation verification, skipping data-track validation.

full_agent: Enlists all concurrent modules working under dynamic self-correction loops.

⚡ Getting Started
1. Prerequisites & Installation
Clone the repository and install the required core packages:

Bash
git clone [https://github.com/your-username/AeroAgent.git](https://github.com/your-username/AeroAgent.git)
cd AeroAgent
pip install langchain-chroma langchain-community sentence-transformers google-generativeai tenacity
2. Configure Environment Keys
Set up your Google AI Studio token as an environment variable:

Bash
export GEMINI_API_KEY="your_api_key_here"
3. Run the Evaluation Pipeline
To sweep across all 30 evaluation test instances across every ablation mode, execute the evaluation suite:

Bash
# Clear any prior cached or corrupted evaluation tracks
rm -rf predictions/*.jsonl

# Execute the complete automated pipeline sweep
python3 -m src.evaluator
📜 Compliance Guidance Rules
The model dynamically shifts its prompt payload and safety token budgets based on the designated question_type:

Factoid Answers: Limited strictly to 1 to 3 concise sentences.

Comparative Answers: Balanced architectural trade-off breakdowns bounded between 100 and 300 words.

Survey Answers: Highly detailed, multi-perspective structural overviews bounded between 250 and 600 words.