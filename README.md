# Multimodal CBSE Study Assistant (Class 12 Physics)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-8A2BE2.svg)](https://qdrant.tech/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Inference-Groq-F55036.svg)](https://groq.com/)

## 2. Executive Summary

The Multimodal CBSE Study Assistant is a textbook-grounded question-answering
system for Class 12 Physics. It retrieves and reasons over educational
material that contains both written explanations and complex visual content,
including diagrams, graphs, equations, tables, and labeled textbook figures.

Instead of treating a textbook as text alone, the system represents both
page-level visual evidence and extracted text in a shared retrieval pipeline.
A student receives:

- A clear, grounded explanation.
- Source and page references for the retrieved evidence.
- The relevant full textbook page images for visual inspection.

The generation prompt treats retrieved textbook evidence as the factual
boundary for textbook-specific claims. This makes unsupported claims visible,
encourages the model to acknowledge insufficient evidence, and provides a
practical safeguard against hallucinated page references or explanations.

## 3. Architecture & Tech Stack

### End-to-end pipeline

```text
Raw textbook PDFs
        |
        v
PDF page-text extraction with pypdf
        |
        +------------------------------+
        |                              |
        v                              v
Text cleaning and page-aware       Full-page rendering
semantic chunking                  with PyMuPDF
        |                              |
        |                              v
        |                       VLM page descriptions
        |                       for diagrams, equations,
        |                       labels, and tables
        |                              |
        +--------------+---------------+
                       v
             Text and page records
                       |
                       v
       Normalized BGE-base embeddings
                       |
                       v
          Qdrant batched UUID upserts
                       |
                       v
              Multimodal retrieval
                       |
                       v
          Grounded Groq answer generation
                       |
                       v
        Streamlit answer, citations,
              and full page images
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
7. Saves detailed per-question results to
   [`data/evaluation/results.json`](data/evaluation/results.json).

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

These observations motivate future techniques such as query decomposition,
hybrid lexical-plus-dense retrieval, page diversification, section-aware
metadata filtering, and reranking. They are not included in the current
vertical slice so that its behavior remains simple and measurable.

## 6. Future Roadmap

### 1. Decouple the retrieval engine from Streamlit

Wrap the core retrieval and generation path in a robust FastAPI backend. This
would make the RAG engine independently testable and reusable by Streamlit,
future web clients, or batch evaluation jobs.

### 2. Containerize and deploy

Package the application and supporting services in containers, then use
available AWS credits for a controlled cloud deployment. The deployment should
include secure environment-variable handling, persistent Qdrant storage, and
basic operational monitoring.

### 3. Expand the textbook

Extend the current Chapter 1 vertical slice to the remaining seven Physics
chapters. The ingestion process is already designed to preserve document,
page, source-file, and record-type metadata so the expanded collection can be
rebuilt deterministically.

### 4. Improve retrieval based on measured failures

Use the current failure report to evaluate:

- Hybrid sparse and dense retrieval.
- Query decomposition for comparative and multi-hop questions.
- Reranking of candidate pages.
- Section and chapter metadata filters.
- Page-level diversification.

The goal is not to obscure the current metric, but to show measurable progress
against the exact failure modes identified in the Chapter 1 benchmark.

## 7. Project Structure

```text
MultiModal_RAG/
├── app/
│   └── streamlit_app.py          # Student-facing UI
├── data/
│   ├── evaluation/              # Benchmark and evaluation reports
│   ├── pages/                   # Rendered textbook pages
│   ├── processed/               # Records and visual metadata
│   └── raw/                     # Source textbook PDFs
├── scripts/
│   ├── evaluate.py              # Retrieval and generation evaluation
│   ├── index.py                 # Qdrant indexing
│   ├── ingest.py                # PDF ingestion
│   └── smoke_test.py            # One-chapter vertical slice
├── src/
│   └── study_assistant/         # Core RAG modules
├── tests/
│   └── test_core.py             # Deterministic unit tests
├── .env.example
├── requirements.txt
└── README.md
```

## 8. Setup and Usage

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

```powershell
streamlit run app\streamlit_app.py
```

### Run tests

```powershell
python -m pytest -q
```

## 9. Limitations

- The current benchmark covers Chapter 1 and is intentionally small.
- Recall is measured against expected page numbers, so semantically useful
  neighboring pages can still count as misses.
- LLM-as-a-judge metrics are secondary evaluation signals and require manual
  review for high-stakes conclusions.
- Groq and Qdrant availability, quotas, and model behavior depend on the
  configured external services.
- The current production path covers Chapter 1; the remaining seven textbook
  chapters are planned for expansion.
