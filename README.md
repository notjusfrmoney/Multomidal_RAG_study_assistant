# Multimodal CBSE Study Assistant

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-8A2BE2)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Groq](https://img.shields.io/badge/LLM-Groq-F55036)

A textbook-grounded study assistant for Class 12 Physics. It retrieves
extracted textbook text together with full-page visual descriptions, then
generates an answer constrained by the retrieved evidence. The project is a
small applied RAG system with a FastAPI backend, Streamlit interface, Qdrant
index, deterministic calculator, and an evaluation suite.

## Demo / Overview

Students can ask textbook questions, ask follow-up questions, compare
concepts, request study guidance, or calculate the electric field of a point
charge. The response includes the answer and, when retrieval is used, the
source file, page number, record type, and available textbook page image.

The repository currently contains a Class 12 Physics Chapter 1 vertical slice:
**Electric Charges and Fields**. It is not a complete CBSE curriculum index.

## Problem

Text-only retrieval is not enough for textbooks that explain ideas through
diagrams, equations, graphs, tables, worked examples, and page layout.
This project keeps both extracted page text and full-page visual evidence in
the searchable knowledge base so that visual and text-based questions can use
the same grounded workflow.

## Key Features

- Textbook-grounded question answering
- Text and full-page visual/page records in Qdrant
- Groq-based query routing and grounded answer generation
- Conversation-aware follow-up rewriting
- Multi-concept query decomposition, bounded to three subqueries
- Evidence sufficiency checks before answer generation
- Bounded retrieval retry when evidence is unusable
- Bounded retries for transient Groq connection/time-out errors
- Deterministic electric-field calculation for a point charge
- Source-file and page-level evidence in API/UI responses
- FastAPI `/query` and `/health` endpoints
- Streamlit student chat interface
- Dockerfiles and Docker Compose configuration
- Deterministic tests and two evaluation runners

## Architecture

```text
Student
   |
   v
Streamlit UI
   |
   v
FastAPI /query
   |
   v
Study Assistant Orchestrator
   |
   +--> Query router
   +--> Follow-up rewriter
   +--> Query decomposer
   +--> Evidence assessment
   +--> Deterministic calculator (calculation questions)
   |
   v
Independent text/page searches
   |
   v
Round-robin merge and deduplication
   |
   v
Qdrant collection
   |
   v
Grounded Groq answer generation
   |
   v
Answer + sources + page images
```

The ingestion path is separate from the online query path:

```text
PDF files
   |
   +--> pypdf page text extraction
   +--> PyMuPDF full-page PNG rendering
   +--> Groq vision descriptions for each page
   |
   v
Text chunks + page records
   |
   v
BAAI/bge-base-en-v1.5 embeddings
   |
   v
Qdrant
```

## Data / Knowledge Base

The current processed material is the Class 12 Physics chapter
`chapter_01_electric_charges_and_fields`. Ingestion creates:

- `text` records containing cleaned page-aware text chunks
- `page` records containing a generated description of the full textbook page
  and its rendered PNG path
- Page number, source file, record type, document ID, and chunk ID metadata

The current ingestion code processes full pages; it does not crop individual
figures into separate image records.

## RAG Pipeline

1. Read textbook PDFs from `data/raw`.
2. Extract page text with `pypdf`.
3. Render each page as a PNG with PyMuPDF.
4. Ask the configured Groq vision model to describe page-level visual content.
5. Create overlapping text chunks and one page record per page.
6. Embed record text with `BAAI/bge-base-en-v1.5`.
7. Index records in a Qdrant collection.
8. Route the student's message into an implemented intent.
9. Rewrite follow-ups and decompose multi-concept questions when needed.
10. Retrieve the configured top-k records for each search query.
11. Merge decomposed results round-robin, remove duplicates, and assess evidence.
12. Retry retrieval once with a normalized textbook-context suffix if evidence
    is unusable.
13. Generate a grounded response or return an insufficient-evidence response.
14. Return source metadata and page images through the API/UI.

## Agentic Query Handling

### Query routing

The router recognizes these intents:

- `casual_chat`
- `textbook_question`
- `follow_up_question`
- `calculation`
- `study_guidance`

Obvious greetings and acknowledgements are handled without an LLM router call.

### Follow-up questions

For example:

```text
Student: What is electric flux?
Follow-up: Explain that again.
```

Recent conversation history is used to rewrite the follow-up into a standalone
retrieval query before searching. The router only uses the recent bounded
history configured in the application; this is not unlimited memory.

### Query decomposition

A multi-concept question can be split into up to three independent retrieval
queries. For example, a comparison of point-charge and dipole field diagrams
can produce separate point-charge and dipole searches. Their results are
interleaved so one subquery does not fill the entire top-k context.

### Evidence checking and retry

Retrieved records are checked for usable text or page-image evidence. The
orchestrator allows a maximum of two retrieval attempts. If usable evidence
still cannot be obtained, it returns an explicit insufficient-evidence answer
instead of pretending that the textbook supports the response.

Main Groq calls also retry transient connection/time-out failures at most twice
with short backoff. Rate-limit errors remain explicit failures.

## Multimodal Retrieval

The index contains extracted text records and full-page visual/page records.
Visual records describe diagrams, equations, labels, tables, and the main
concept on a complete textbook page. This supports questions where the useful
evidence is in page context or a diagram rather than in a clean text-only
paragraph.

The current system uses full textbook page images, not cropped figure
retrieval. The Streamlit UI displays returned page images when they are
available locally.

## Numerical Questions

The deterministic calculator currently supports electric field magnitude for a
point charge:

```text
E = k |q| / r²
```

It parses charge units including `C`, `mC`, `μC`, `µC`, `uC`, and `nC`, and
distance units including `m`, `cm`, and `km`. It returns a structured success
or failure result and does not execute arbitrary code or claim to support every
Physics formula.

## API, UI, and Docker

### Local setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GROQ_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY` in `.env`. Model,
collection, and top-k settings can also be configured there.

### Build the local knowledge base

Place the textbook PDF under `data/raw`, then run:

```powershell
python scripts\ingest.py --chapter class_12_physics_part01\chapter_01_electric_charges_and_fields.pdf
python scripts\index.py
```

The ingestion step writes processed records and page images under `data`.
Those generated textbook artifacts are local data and are excluded from the
public repository by `.gitignore`.

### Run locally

Start the API:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
streamlit run app\streamlit_app.py
```

The API is available at `http://localhost:8000`; the UI is available at
`http://localhost:8501`.

### Run with Docker Compose

```powershell
docker compose up --build
```

Compose starts separate API and UI containers. The UI uses
`http://api:8000` for its internal API URL.

## Evaluation

The project includes two evaluation runners:

- `scripts/evaluate.py` runs the original five-question Chapter 1 benchmark
  from `data/evaluation/benchmark_chapter_01.json`.
- `scripts/evaluate_agentic.py` runs the 24-case benchmark in
  `evaluation/benchmark.json`, covering routing, follow-ups, decomposition,
  visual questions, calculations, insufficient evidence, citations, and
  end-to-end success.

Run the deterministic test suite:

```powershell
python -m pytest -q
```

Run the baseline evaluation:

```powershell
python -m scripts.evaluate
```

Run the agentic evaluation:

```powershell
python -m scripts.evaluate_agentic
```

### Evaluation methodology

The agentic evaluator uses a **custom repository-local evaluation harness**;
it does not use RAGAS or DeepEval:

```text
24 benchmark cases
        ↓
run routing, retrieval, orchestration, and answer generation
        ↓
retrieval and citation checks
        ↓
agentic behavior checks
        ↓
answer scoring where a reference answer exists
        ↓
aggregate metrics
```

Retrieval metrics use the cases with expected source pages (13 cases in the
stored run). Recall@k is the fraction of expected pages found in the first k
retrieved pages; Hit@k records whether at least one expected page appears in
the first k. Citation Accuracy uses the same 13 cases and checks whether the
answer cites an expected page. Follow-up Resolution and Contextual Retrieval
Success use the 3 follow-up cases. Decomposition Decision Accuracy and
Concept Coverage use the multi-concept cases.

Correctness and Groundedness are LLM-judge scores from 1 to 5, averaged only
for the 8 cases in the stored run that contain a reference answer. They are
not scores for all 24 cases. End-to-End Success is a case-level boolean over
all 24 cases and includes routing, retrieval, abstention, calculation,
decomposition, answer, and citation requirements where applicable.

The latest completed 24-case agentic run produced:

| Metric | Result | Evaluation scope |
|---|---:|---|
| Recall@3 | 53.8% | 13 cases with expected source pages |
| Recall@5 | 69.2% | 13 cases with expected source pages |
| Hit@3 | 61.5% | 13 cases with expected source pages |
| Hit@5 | 69.2% | 13 cases with expected source pages |
| Citation Accuracy | 69.2% | 13 cases with expected source pages |
| Follow-up Resolution | 100.0% | 3 follow-up cases |
| Contextual Retrieval Success | 100.0% | 3 follow-up cases |
| Decomposition Decision Accuracy | 100.0% | Multi-concept cases |
| Concept Coverage | 100.0% | Multi-concept cases |
| Answer Correctness | 5.00 / 5 | 8 cases with reference answers; LLM judge |
| Answer Groundedness | 5.00 / 5 | 8 cases with reference answers; LLM judge |
| Calculation Accuracy | 100.0% | Numerical cases with expected results |
| Unsupported-query abstention success | 100.0% | 3 insufficient-evidence cases; recomputed offline |
| End-to-End Success Rate | 83.3% | 20/24 cases; recomputed offline |

Unsupported-query abstention success means that, for an insufficient-evidence
case, the generated answer explicitly states that the available evidence is
insufficient. Its denominator is the 3 insufficient-evidence cases; it is not
an unsupported-answer rate.

The retrieval, citation, follow-up, decomposition, calculation, and LLM-judge
values above are directly reusable from the stored 24-case benchmark run.
Abstention success and End-to-End Success were recomputed offline from those
same stored outputs using the corrected evaluator logic; no model or retrieval
calls were made. A fresh live evaluation is still required to measure the
current retrieval implementation, including the recent lightweight reranking
change.

These are local evaluation results for the current Chapter 1 benchmark, not
claims about the entire CBSE curriculum. The benchmark has 24 structured
cases, which is useful for validating the current implementation but is too
small for statistically robust production-level estimates. A larger benchmark
would be needed for stronger confidence. Detailed reports are written to
`data/evaluation/results.json` and `data/evaluation/agentic_results.json`.

Each online query now records `total_latency_ms` in the orchestrator debug
payload using Python's standard `time.perf_counter()`. The stored benchmark
artifact also contains per-case `latency_ms`, but those values include the
configured live evaluation path and should not be treated as portable
performance guarantees.

### Evaluation limitations

- The benchmark covers one Class 12 Physics chapter and 24 structured cases.
- Stored results are a snapshot of one configured run.
- Correctness and groundedness use an LLM judge, not human evaluation.
- Latency depends on Groq, Qdrant, network conditions, and local configuration.
- A broader multi-chapter benchmark and repeated live runs would be needed
  for stronger statistical confidence.

## Tests

The test suite covers:

- Text cleaning, chunking, IDs, and citations
- Calculator parsing and deterministic results
- Query routing, follow-up rewriting, decomposition, and orchestration
- Transient Groq retry behavior
- Evaluation metrics and benchmark provenance
- Diversified merge ordering, bounds, and deduplication

The current local suite has **43 passing tests**.

## Project Structure

```text
MultiModal_RAG/
├── api/main.py                    # FastAPI application
├── app/streamlit_app.py           # Streamlit chat UI
├── data/                          # Local raw/processed/evaluation data
├── evaluation/benchmark.json      # Agentic benchmark
├── scripts/
│   ├── ingest.py                  # PDF ingestion
│   ├── index.py                   # Qdrant indexing
│   ├── evaluate.py                # Baseline evaluation
│   ├── evaluate_agentic.py        # Agentic evaluation
│   └── smoke_test.py              # Chapter vertical-slice check
├── src/study_assistant/
│   ├── agent.py                   # Routing, rewriting, decomposition
│   ├── calculator.py              # Electric-field calculator
│   ├── chunking.py                # Text cleanup and chunk records
│   ├── embeddings.py              # Sentence-transformer embeddings
│   ├── generation.py              # Grounded answer generation
│   ├── ingest.py                  # PDF/text/page processing
│   ├── orchestrator.py            # End-to-end query workflow
│   ├── retrieval.py               # Qdrant search
│   └── visual_understanding.py    # Full-page visual descriptions
├── tests/
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

## Configuration and Safety

Copy `.env.example` to `.env` and keep credentials local. The repository
expects Groq and Qdrant credentials through environment variables and does not
store them in source files. Do not commit `.env`, API keys, downloaded
textbooks, generated page images, processed records, or local evaluation
artifacts.
