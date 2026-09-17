# Multimodal CBSE Study Assistant (Class 12 Physics)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-8A2BE2.svg)](https://qdrant.tech/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Inference-Groq-F55036.svg)](https://groq.com/)

## 2. Executive Summary

The Multimodal CBSE Study Assistant is a textbook-grounded conversational
agent for Class 12 Physics. It retrieves and reasons over educational material
that contains both written explanations and complex visual content, including
diagrams, graphs, equations, tables, and labeled textbook figures.

Instead of treating a textbook as text alone, the system represents both
page-level visual evidence and extracted text in a shared retrieval pipeline.
A student receives:

- A clear, grounded explanation.
- Source and page references for the retrieved evidence.
- The relevant full textbook page images for visual inspection.

The generation prompt treats retrieved textbook evidence as the factual
boundary for textbook-specific claims. Evidence assessment, bounded retry, and
an honest insufficient-evidence response help prevent unsupported textbook
answers. Conversational routing and follow-up rewriting allow the same system
to support both study dialogue and textbook-grounded questions.

## 3. Architecture & Tech Stack

### End-to-end pipeline

```text
Student
   |
   v
Streamlit chat UI
   |
   v
FastAPI /query
   |
   v
Deterministic obvious-casual routing
   |
   v
LLM query router
   |
   v
Conversation-aware query rewriting
   |
   v
Optional query decomposition
   |
   v
Multimodal Qdrant retrieval
   |
   v
Evidence assessment
   |
   v
Bounded retrieval retry
   |
   v
Deterministic calculator when applicable
   |
   v
Grounded Groq generation
   |
   v
Answer + citations + relevant textbook page images
```

The ingestion and indexing path produces the multimodal evidence used by the
agent:

```text
Raw textbook PDFs
        |
        +--> Page text extraction, cleaning, and page-aware chunking
        |
        +--> Full-page rendering with PyMuPDF
                    |
                    +--> VLM descriptions for diagrams, equations,
                         tables, labels, and other visual evidence
        |
        v
Text and page records
        |
        v
Normalized BGE embeddings
        |
        v
Qdrant collection
```

### Technology choices

- **VLM and generation:** `qwen/qwen3.8-27b` through Groq. The model ID is
  configurable through `VLM_MODEL` and `GENERATION_MODEL`; the current
  default provides a capable multimodal and reasoning path while keeping the
  project aligned with a free-to-use Groq development setup.
- **Embeddings:** `BAAI/bge-base-en-v1.5`, producing verified
  **768-dimensional** normalized vectors.
- **Vector database:** Qdrant with one collection containing two logical
  record types: text chunks and page records. Records use deterministic UUIDs
  and are uploaded in batches to avoid oversized requests.
- **PDF processing:** `pypdf` for page text extraction and PyMuPDF for
  consistent full-page PNG rendering.
- **User interface:** Streamlit, focused on a question box, grounded answer,
  citations, and relevant textbook pages.
- **Provider configuration:** Groq and Qdrant credentials are loaded from
  environment variables. Secrets are not stored in source code.

### Implemented capabilities

#### Multimodal RAG

- Page-level text extraction.
- Text cleaning and page-aware chunking.
- Full-page rendering with PyMuPDF.
- VLM-generated page descriptions for diagrams, equations, and tables.
- Text and page records with page metadata.
- BGE embeddings and Qdrant retrieval.
- Grounded answer generation.
- Page and source citations.
- Relevant full-textbook-page images in the Streamlit chat.

#### Agentic layer

- Deterministic handling of obvious standalone greetings and acknowledgements.
- LLM-based query routing.
- Supported intents: `casual_chat`, `textbook_question`,
  `follow_up_question`, `calculation`, and `study_guidance`.
- Conversation-aware follow-up query rewriting.
- Query decomposition for multi-concept questions.
- Bounded retrieval attempts with lightweight evidence-quality assessment.
- Safe insufficient-evidence responses when usable textbook evidence is not
  found.
- An orchestration layer that separates application workflow from API
  transport.

#### Deterministic numerical tool

The current calculator intentionally supports one operation:

- Electric field due to a point charge using `E = k|q| / r²`.
- Charge units: `C`, `mC`, `μC`, `µC`, `uC`, and `nC`.
- Distance units: `m`, `cm`, and `km`.
- Deterministic Python arithmetic rather than arbitrary code execution.
- Safe failure for missing or invalid inputs.

It does not claim to support arbitrary Physics formulas.

### Engineering philosophy

The implementation relies on clean, modular, vanilla Python with small
focused modules and straightforward data flow. It deliberately avoids
unnecessary service layers, factories, and framework-specific abstractions.

Tools such as LangChain and LangGraph are industry standards and can be
appropriate for larger systems. This project instead builds the core pipeline
from scratch to demonstrate first-principles understanding of RAG mechanics,
PDF and page processing, chunking, embeddings, multimodal records, retrieval,
grounded generation, citations, and deterministic metadata handling.

## 4. Evaluation & Metrics

The project includes a custom evaluation loop implemented in
[`scripts/evaluate.py`](scripts/evaluate.py), without DeepEval, Ragas,
LangChain, or another heavy evaluation framework.

For each benchmark question, the evaluator:

1. Loads the question, gold answer, and expected source pages.
2. Retrieves the top five records from Qdrant.
3. Calculates a boolean `Recall@5` value by checking whether an expected page
   appears among the retrieved page numbers.
4. Generates a grounded answer using the configured Groq model.
5. Calls Groq with a separate strict judge prompt.
6. Scores answer correctness and groundedness from 1 to 5.
7. Saves detailed per-question results locally to
   `data/evaluation/results.json`. Generated evaluation artifacts are ignored
   so they are not uploaded with the public source repository.

The current Chapter 1 benchmark results are:

| Metric | Result | Interpretation |
|---|---:|---|
| **Average Correctness** | **5.000** | Perfect factual alignment when relevant context is found |
| **Average Groundedness** | **4.200** | Strong adherence to the supplied textbook evidence |
| **Average Recall@5** | **0.400** | Two of five benchmark questions retrieved at least one expected page |

LLM-as-a-judge scores are useful diagnostic signals, not absolute truth. The
benchmark and report should therefore be interpreted alongside manual
inspection of retrieved evidence and generated answers.

Run the evaluation with:

```powershell
python -m scripts.evaluate
```

The practical agentic evaluation uses the richer 24-case benchmark in
[`evaluation/benchmark.json`](evaluation/benchmark.json). It covers direct
textbook questions, follow-ups, multi-concept questions, casual conversation,
study guidance, numerical requests, visual questions, and insufficient
evidence. The evaluator implements:

- Intent Accuracy
- Retrieval Decision Accuracy
- Unnecessary Retrieval Rate
- Hit@3 and Hit@5
- Recall@3 and Recall@5
- Follow-up Resolution Accuracy
- Contextual Retrieval Success
- Decomposition Decision Accuracy
- Concept Coverage
- Answer Correctness
- Groundedness
- Citation Accuracy
- Calculation Accuracy
- Unsupported Answer Rate
- End-to-End Success Rate

```powershell
python -m scripts.evaluate_agentic
```

The agentic evaluator requires usable Groq quota for routing, answer
generation, and optional judging. It writes detailed results to the ignored
path `data/evaluation/agentic_results.json`.

The evaluation framework is implemented and locally validated with **29
passing tests**. The evaluator was successfully executed, and a small
evaluator bug was fixed during the live run. Live end-to-end quality metrics
are pending completion of the benchmark run because the configured Groq
service reached its daily token quota. The evaluation framework itself is
implemented and locally validated. This is an external provider-quota
limitation, not a change to the application architecture or benchmark.

## 5. Failure Analysis: The Engineering Highlight

The **40% Recall@5** result is not simply a product failure. It exposes a
well-known limitation of standard dense vector retrieval and provides a
concrete direction for the next retrieval iteration.

The current system often understands the topic but does not always return the
exact page label used by the benchmark. This distinction matters in a
textbook assistant because adjacent pages can be semantically useful while
still failing strict source-page recall.

### Near misses

For the uniformly charged spherical shell question, Qdrant returned relevant
neighboring material from pages **35, 36, and 40**, but the benchmark expected
page **39**. The retrieved content remained in the correct conceptual region:
it discussed Gaussian surfaces, spherical shells, and the electric field.
However, dense similarity ranked adjacent explanations above the exact
benchmark page.

Similarly, the quantization question retrieved a related page discussing
quantization but missed the benchmark's expected page 8. This shows that
semantic proximity and exact provenance recall are related but different
objectives.

### Multi-hop and comparative concepts

The point-charge-versus-dipole question demonstrates another limitation. A
single-vector query must represent two related but distinct concepts at once:

1. Radial field lines from a single positive point charge.
2. Field lines beginning at a positive charge and ending at a negative charge
   in an electric dipole.

The retriever returned nearby pages about electric-field and dipole concepts,
but it did not retrieve either expected page 18 or page 24 within the top five.
The query's combined meaning was effectively placed between multiple regions
of the latent space rather than producing two deliberate evidence searches.

These observations motivate future techniques such as hybrid
lexical-plus-dense retrieval, page diversification, section-aware metadata
filtering, and reranking. Query decomposition is already implemented in the
current agentic layer; these additional retrieval improvements remain outside
the current vertical slice so that its behavior stays measurable.

## 6. Service Architecture and Deployment

The core RAG engine is exposed through a thin FastAPI service. Streamlit acts
as the presentation layer and sends questions to the API rather than importing
retrieval and generation code directly.

Available endpoints:

- `GET /health` checks API and Qdrant availability.
- `POST /query` routes the conversation, optionally retrieves evidence,
  generates either a conversational or grounded answer, and returns
  operational metadata, source citations, and page references.

The project includes separate Docker images for the API and UI, coordinated by
Docker Compose. The UI uses `API_URL=http://api:8000` inside the Compose
network, while local non-container runs default to `http://localhost:8000`.

### Run with FastAPI and Streamlit locally

Start the API:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, start the UI:

```powershell
streamlit run app\streamlit_app.py
```

### Run with Docker Compose

```powershell
docker compose up --build
```

The services are then available at:

- FastAPI: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

The Compose configuration loads local environment values from `.env` without
copying that file into either image.

## 7. Public Repository Safety

The repository is prepared for public GitHub publication:

- `.env` is ignored and must contain local secrets only.
- `.env.example` documents variable names without secret values.
- Generated textbook data, rendered pages, processed records, and local Qdrant
  storage are excluded through `.gitignore`.
- Python caches, test caches, virtual environments, and IDE settings are
  excluded.

Before publishing, verify the staged file list and scan for credentials:

```powershell
git status --short
git ls-files
```

Never commit API keys, Qdrant credentials, downloaded textbook artifacts, or
local database files.

## 8. Future Roadmap

### 1. Container deployment

Use available AWS credits for a controlled cloud deployment with secure secret
management, persistent Qdrant storage, health checks, and basic monitoring.

### 2. Expand the textbook

Extend the current Chapter 1 vertical slice to the remaining seven Physics
chapters. The ingestion process preserves document, page, source-file, and
record-type metadata so the expanded collection can be rebuilt deterministically.

### 3. Improve retrieval based on measured failures

Use the current failure report to evaluate:

- Hybrid sparse and dense retrieval.
- Reranking of candidate pages.
- Section and chapter metadata filters.
- Page-level diversification.
- More robust evaluation of multi-concept decomposition and evidence quality.

The goal is not to obscure the current metric, but to show measurable progress
against the exact failure modes identified in the Chapter 1 benchmark.

## 9. Project Structure

```text
MultiModal_RAG/
├── api/
│   └── main.py                   # FastAPI backend
├── app/
│   └── streamlit_app.py          # Student-facing UI
├── data/                         # Local, ignored textbook artifacts
├── scripts/
│   ├── evaluate.py              # Retrieval and generation evaluation
│   ├── evaluate_agentic.py      # Agentic evaluation runner
│   ├── evaluation_metrics.py    # Agentic evaluation metrics
│   ├── index.py                 # Qdrant indexing
│   ├── ingest.py                # PDF ingestion
│   └── smoke_test.py            # One-chapter vertical slice
├── src/
│   └── study_assistant/         # RAG, agent, orchestration, and calculator
├── tests/
│   ├── test_core.py             # Core deterministic tests
│   ├── test_agent.py            # Agent and orchestration tests
│   ├── test_calculator.py       # Calculator tests
│   └── test_evaluation.py       # Evaluation metric tests
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 10. Setup and Usage

### Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Configure environment

Copy `.env.example` to `.env` and configure:

```text
GROQ_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
QDRANT_COLLECTION=student_physics
TOP_K=5
GENERATION_MODEL=qwen/qwen3.8-27b
VLM_MODEL=qwen/qwen3.8-27b
```

Never commit `.env` or expose its values.

### Run the Chapter 1 vertical slice

```powershell
python -m scripts.smoke_test --chapter class_12_physics_part01\chapter_01_electric_charges_and_fields.pdf
```

### Ingest and index

```powershell
python scripts\ingest.py
python scripts\index.py
```

### Start the application

For the service-oriented local setup, start FastAPI first:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Then start Streamlit in a second terminal:

```powershell
streamlit run app\streamlit_app.py
```

Alternatively, launch both services with Docker Compose:

```powershell
docker compose up --build
```

### Run tests

```powershell
python -m pytest -q
```

## 11. Limitations

- The current benchmark covers Chapter 1 and is intentionally small.
- Recall is measured against expected page numbers, so semantically useful
  neighboring pages can still count as misses.
- LLM-as-a-judge metrics are secondary evaluation signals and require manual
  review for high-stakes conclusions.
- Groq and Qdrant availability, quotas, and model behavior depend on the
  configured external services.
- The current production path covers Chapter 1; the remaining seven textbook
  chapters are planned for expansion.
