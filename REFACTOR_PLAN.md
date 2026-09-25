# Autonomous Refactor Plan: Multimodal CBSE Study Assistant → LangChain + LangGraph

**How this file is meant to be used:** commit it to the repo root as
`REFACTOR_PLAN.md` before doing anything else. Give the agent the master
prompt (Appendix C) once. It reads this file, executes phases 0→11 in order,
verifies itself against each phase's **Phase gate** using the literal
commands given, commits on pass, and stops with a written blocker report on
failure. You are not watching it run — your checkpoints are the git log,
`REFACTOR_BLOCKERS.md` if it appears, and the final PR diff.

---

## Autonomous execution contract

The agent follows this for every phase, no exceptions:

1. Read the phase's Goal and Steps. Execute them.
2. Run the exact commands in that phase's **Phase gate**.
3. Gate passes → `git add` the changed files, commit with the given message
   pattern, move to the next phase.
4. Gate fails → fix and re-verify, up to **3 attempts** for that phase.
5. Still failing after 3 attempts → STOP. Do not touch later phases. Append
   an entry to `REFACTOR_BLOCKERS.md` (create if missing): which phase, what
   was tried, the exact failure output, a suggested next step. End the
   session there. Do not push past a failing gate to keep momentum.
6. All work happens on branch `langgraph-refactor`. Never commit to
   `main`/`master`. Never force-push. Never touch `.env`, credentials, or
   files under `data/`.
7. Final step after Phase 11 (or after a blocker): push the branch and stop.
   **Do not merge into main autonomously** — leave that for a one-time human
   review of the PR diff. (If you genuinely want zero human touchpoints at
   all, delete this line — but for a portfolio project, this is the cheapest
   insurance you'll ever buy.)

---

## Phase 0 — Branch, baseline, commit this plan

**Steps:**
1. `git checkout -b langgraph-refactor`
2. `python -m pytest -q > baseline_tests.txt`
3. Copy `data/evaluation/results.json` and `data/evaluation/agentic_results.json`
   into `baseline_results/`.
4. `git add REFACTOR_PLAN.md && git commit -m "Add autonomous refactor plan"`

**Phase gate:**
```
test -f baseline_tests.txt && test -d baseline_results \
  && grep -q "passed" baseline_tests.txt && echo GATE_PASS || echo GATE_FAIL
```

**Commit message:** `Phase 0: baseline captured, refactor branch created`

---

## Phase 1 — Add dependencies

**Files:** `requirements.txt`

**Steps:** add `langchain`, `langchain-core`, `langchain-groq`,
`langchain-qdrant`, `langchain-huggingface`, `langgraph`. Install. Pin exact
installed versions back into `requirements.txt`.

**Phase gate:**
```
pip install -r requirements.txt && \
python -c "import langchain, langgraph, langchain_groq, langchain_qdrant, langchain_huggingface" \
  && echo GATE_PASS || echo GATE_FAIL
```

**Commit message:** `Phase 1: add LangChain/LangGraph dependencies`

---

## Phase 2 — Embeddings + vector store → LangChain

**Files:** `src/study_assistant/embeddings.py`, `src/study_assistant/retrieval.py`

**Steps:**
1. `embeddings.py`: wrap `BAAI/bge-base-en-v1.5` in
   `langchain_huggingface.HuggingFaceEmbeddings`, same function signature.
2. `retrieval.py`: replace raw `qdrant-client` search with
   `langchain_qdrant.QdrantVectorStore` against the existing collection.
   Preserve every metadata field (`record_type`, `page_number`,
   `source_file`, `doc_id`, `chunk_id`) in each `Document.metadata`.
3. Leave `scripts/index.py` (ingestion) on raw `qdrant-client` — untouched.

**Phase gate:** full test suite passes, AND a fixed known-answer query
returns the expected page in its top-5 (use one of the benchmark's
page-grounded cases from `evaluation/benchmark.json` as the fixed query —
pick one with an unambiguous expected page).
```
python -m pytest -q && \
python -c "
from src.study_assistant.retrieval import search  # adjust import to actual function name
results = search('<fixed benchmark query text>', top_k=5)
pages = [r['metadata']['page_number'] for r in results]
assert <expected_page> in pages, f'expected page missing: {pages}'
print('GATE_PASS')
"
```

**Commit message:** `Phase 2: embeddings/vector store onto LangChain (HuggingFaceEmbeddings, QdrantVectorStore)`

---

## Phase 3 — Groq calls → LangChain (`ChatGroq`)

**Files:** `src/study_assistant/agent.py`, `src/study_assistant/generation.py`

**Steps:**
1. Replace raw Groq client calls with `langchain_groq.ChatGroq`.
2. Move existing prompt strings into `ChatPromptTemplate.from_messages([...])`
   without changing wording.
3. Router: replace custom text parsing with
   `llm.with_structured_output(RouterIntent)` — a Pydantic model constrained
   to the five existing intents.
4. Retry: replace the hand-rolled bounded retry with `.with_retry()`,
   retrying only connection/timeout exceptions, explicitly excluding
   rate-limit exceptions (must still fail immediately), matching the
   existing max-2-attempts bound.
5. Leave the no-LLM-call greeting short-circuit untouched.

**Phase gate:** full test suite passes, AND a scripted check of 5 fixed
inputs (one per intent: casual_chat, textbook_question, follow_up_question,
calculation, study_guidance) all route to the correct intent via the new
structured router.
```
python -m pytest -q && \
python -c "
from src.study_assistant.agent import route  # adjust to actual function name
cases = {
    'hey there': 'casual_chat',
    'What is electric flux?': 'textbook_question',
    'Explain that again': 'follow_up_question',
    'Calculate the field at 2m from a 5uC charge': 'calculation',
    'How should I study this chapter?': 'study_guidance',
}
for msg, expected in cases.items():
    got = route(msg)
    assert got == expected, f'{msg!r} -> {got}, expected {expected}'
print('GATE_PASS')
"
```

**Commit message:** `Phase 3: Groq calls onto ChatGroq with structured routing output and with_retry`

---

## Phase 4 — LangGraph `State` schema

**Files:** new `src/study_assistant/graph_state.py`

**Steps:** define a `TypedDict StudyAssistantState` covering every field
currently threaded through `orchestrator.py`: `user_message`,
`conversation_history`, `intent`, `rewritten_query`, `subqueries`,
`candidates`, `reranked_results`, `evidence_sufficient`, `retry_count`,
`calculation_result`, `final_answer`, `sources`. Verify against the actual
current orchestrator code — add any field that's missing.

**Phase gate:**
```
python -c "from src.study_assistant.graph_state import StudyAssistantState; print('GATE_PASS')"
```

**Commit message:** `Phase 4: define LangGraph state schema`

---

## Phase 5 — Wrap existing logic as graph nodes

**Files:** new `src/study_assistant/graph_nodes.py`

**Steps:** one thin node function per stage — `route_node`,
`rewrite_followup_node`, `decompose_node`, `retrieve_node`,
`rerank_merge_node`, `evidence_check_node`, `calculate_node`,
`generate_node` — each calling the **existing, unmodified** logic and
returning only the state keys it updates. Do not rewrite the reranking,
dedup, or calculator algorithms.

**Phase gate:** every node importable and callable with a minimal fake
state without raising.
```
python -c "
from src.study_assistant.graph_nodes import (
    route_node, rewrite_followup_node, decompose_node, retrieve_node,
    rerank_merge_node, evidence_check_node, calculate_node, generate_node,
)
print('GATE_PASS')
"
```

**Commit message:** `Phase 5: wrap existing orchestration logic as LangGraph nodes`

---

## Phase 6 — Wire the `StateGraph`

**Files:** new `src/study_assistant/graph.py`

**Steps:** build the graph per this control flow —
`route` → (casual_chat/study_guidance → `generate`; calculation →
`calculate`; follow_up_question → `rewrite_followup` → `retrieve`;
textbook_question → `decompose` → `retrieve`) → `rerank_merge` →
`evidence_check` → (sufficient → `generate`; insufficient and
`retry_count < 2` → back to `retrieve`; insufficient and retries exhausted →
`generate` with an insufficient-evidence response). Match the existing
max-2-retrieval-attempts bound exactly. No checkpointer/memory saver —
history is passed in from outside, matching current behavior.

**Phase gate:** graph compiles, and invoking it on 5 canned initial states
(one per intent) completes without raising, and a case engineered to always
report insufficient evidence terminates instead of looping forever (retry
count caps at 2).
```
python -c "
from src.study_assistant.graph import graph
# one canned initial state per intent, plus one forced-insufficient-evidence case
for state in FIVE_CANNED_STATES_PLUS_ONE_FORCED_INSUFFICIENT:
    result = graph.invoke(state)
    assert result['retry_count'] <= 2
print('GATE_PASS')
"
```
(Agent: construct `FIVE_CANNED_STATES_PLUS_ONE_FORCED_INSUFFICIENT` yourself
based on the actual `StudyAssistantState` fields.)

**Commit message:** `Phase 6: wire LangGraph StateGraph with routing and evidence-retry loop`

---

## Phase 7 — Integrate into orchestrator / FastAPI

**Files:** `src/study_assistant/orchestrator.py`

**Steps:** replace the orchestrator's internals with building the initial
`StudyAssistantState` and calling `graph.invoke(...)`. Keep the function's
external signature and return shape byte-for-byte identical so
`api/main.py` and `app/streamlit_app.py` need zero edits.

**Phase gate:** full test suite passes, FastAPI app starts, and `/query` +
`/health` return 200 with the expected response shape on a live local call.
```
python -m pytest -q && \
uvicorn api.main:app --port 8123 &
sleep 3
curl -sf http://localhost:8123/health && \
curl -sf -X POST http://localhost:8123/query -H "Content-Type: application/json" \
  -d '{"message": "What is electric flux?"}' | python -c "import sys,json; d=json.load(sys.stdin); assert 'answer' in d and 'sources' in d" \
  && echo GATE_PASS || echo GATE_FAIL
kill %1
```

**Commit message:** `Phase 7: orchestrator invokes compiled LangGraph graph`

---

## Phase 8 — Regression testing & evaluation (the real quality gate)

**Steps:**
1. `python -m pytest -q` — update mocks that reference old internal
   structure, but never loosen an assertion just to make it pass.
2. `python -m scripts.evaluate` and `python -m scripts.evaluate_agentic`.
3. Compare against `baseline_results/agentic_results.json`.

**Phase gate:** all tests pass, AND End-to-End Success is no more than 10
percentage points below baseline (baseline was 70.8%, so ≥ 60.8%), AND no
previously-working category (routing, follow-up resolution, decomposition,
calculation) drops to 0%.
```
python -m pytest -q && python -m scripts.evaluate_agentic && \
python -c "
import json
new = json.load(open('data/evaluation/agentic_results.json'))
base = json.load(open('baseline_results/agentic_results.json'))
# adjust key paths to actual JSON structure
assert new['end_to_end_success'] >= base['end_to_end_success'] - 10
print('GATE_PASS')
"
```
If this gate fails, treat it as a real regression to fix, not noise —
this is the phase where a bad refactor actually gets caught.

**Commit message:** `Phase 8: regression-tested against baseline eval`

---

## Phase 9 — Cleanup

**Steps:** delete dead code (old retry loops, old router parsing, unused raw
client imports). `scripts/index.py` keeping raw `qdrant-client` is
intentional, not dead code.

**Phase gate:**
```
! grep -rE "groq\.Client\(|QdrantClient\(" src/study_assistant/*.py | grep -v "src/study_assistant/embeddings.py" \
  && echo GATE_PASS || echo GATE_FAIL
```
(Adjust the grep to whatever your actual raw-client call sites looked like —
the point is: nothing outside ingestion still constructs a raw client.)

**Commit message:** `Phase 9: remove dead pre-refactor code`

---

## Phase 10 — Update README

**Steps:** update Architecture (show the graph/nodes/retry edge), Agentic
Query Handling (describe it in graph terms), Key Features (LangGraph
orchestration, LangChain LLM/vector calls), add LangChain/LangGraph badges.
**Explicitly state what's still custom** — the lexical+dense reranker, the
round-robin merge/dedup, the calculator — these are not framework features,
say so. If Phase 8 numbers moved, update the results table with the new run.

**Phase gate:** (this one is inherently subjective — the agent should verify
mechanically that no stale claims remain, then move on without waiting for
human sign-off)
```
grep -qi "langgraph" README.md && grep -qi "langchain" README.md && echo GATE_PASS || echo GATE_FAIL
```

**Commit message:** `Phase 10: update README for LangGraph/LangChain architecture`

---

## Phase 11 — Push (no autonomous merge)

**Steps:** confirm one commit per phase exists, push the branch, open a PR
if `gh` CLI is available.

**Phase gate:**
```
git log --oneline langgraph-refactor | wc -l  # expect ~11 commits
git push -u origin langgraph-refactor && echo GATE_PASS || echo GATE_FAIL
```

**Commit message:** n/a — this phase pushes, it doesn't commit further.

**Stop here.** Do not merge into `main`. Write a final summary (phases
completed, any blocker file present, link to branch/PR) and end the session.

---

## Appendix A — requirements.txt additions
```
langchain
langchain-core
langchain-groq
langchain-qdrant
langchain-huggingface
langgraph
```

## Appendix B — the honest story (resume/interview, not the README)
> "I built the RAG pipeline and orchestration by hand first, then moved
> orchestration onto LangGraph and standardized the LLM/vector-store calls
> onto LangChain once the logic was stable. The reranking, merge/dedup, and
> calculator stayed custom — the frameworks didn't provide those, so there
> was no reason to force them in."

## Appendix C — master prompt (give this to the agent, once)

```
Read REFACTOR_PLAN.md in this repo's root and follow it exactly, including
the "Autonomous execution contract" section at the top.

Execute Phase 0 through Phase 11 in order. For each phase: do the steps,
then run that phase's exact Phase gate commands. If the gate passes, commit
with the given message and move on. If it fails, fix and retry up to 3
times. If still failing after 3 attempts, stop, write a full entry to
REFACTOR_BLOCKERS.md (create it if it doesn't exist) describing the phase,
what you tried, the exact failure output, and a suggested next step — then
end the session without touching later phases.

Work only on the langgraph-refactor branch. Never commit to main. Never
force-push. Never touch .env, credentials, or files under data/. Do not
rewrite the reranking, dedup, or calculator algorithms — only wrap them.

After Phase 11 (or after writing a blocker report), stop. Do not merge into
main. Give me a final summary: which phases completed, whether a blocker
file exists, and the branch/PR to review.
```
