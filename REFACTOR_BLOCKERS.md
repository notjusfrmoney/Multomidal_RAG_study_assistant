# Refactor blockers

## Phase 8 — Regression testing and evaluation

### What was attempted

- `python -m pytest -q` passed: `46 passed, 2 warnings`.
- The agentic evaluator was run three times with `scripts.evaluate_agentic.RESULTS_PATH`
  redirected to a temporary file outside `data/`, because the user instruction
  forbids modifying files under `data/`.
- The regular evaluator was not run because it also writes directly to
  `data/evaluation/results.json`; running it as specified would violate the
  protected-data rule.
- The three agentic evaluation attempts were made after waiting 10 seconds and
  65 seconds between retries.

### Exact failure output

Each agentic attempt failed while routing the first benchmark item with a Groq
rate-limit response. The final attempt reported:

```text
groq.RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached
for model `qwen/qwen3.8-27b` in organization `org_01m0cxe5qhemw9jg0523xrys0h`
service tier `on_demand` on input tokens per minute (ITPM): Limit 7000, Used
6635, Requested 538. Please try again in 1.482857142s.', 'type':
'rate_limit_exceeded'}}

RuntimeError: Groq rate limit reached while processing the structured query
request. Please retry later.
```

The first attempt failed similarly on output-token rate limiting:

```text
groq.RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached
for model `qwen/qwen3.8-27b` ... output tokens per minute (OTPM): Limit 1000,
Used 753, Requested 311. Please try again in 3.84s.', 'type':
'rate_limit_exceeded'}}
```

### Suggested next step

Wait for the Groq quota window to reset or use an evaluation-capable Groq
model/API quota, then rerun both evaluators with their reports redirected or
otherwise reviewed so that protected `data/` files remain unchanged. Compare
the resulting agentic end-to-end success rate and category metrics against
`baseline_results/agentic_results.json`, then resume at Phase 8 before
continuing to Phase 9.

## Phase 8 — Resumed evaluation attempts (2026-09-26)

### What was attempted

- Confirmed the branch was `langgraph-refactor` and resumed at Phase 8.
- `python -m pytest -q` passed: `46 passed, 2 warnings`.
- Restored the prior 200-token completion bound for structured routing in
  `src/study_assistant/agent.py`; the test suite passed after that adjustment.
- A single structured routing request succeeded after a 65-second cooldown.
- Ran `scripts.evaluate.evaluate()` with its report redirected outside `data/`;
  it completed all 5 standard evaluation cases successfully.
- Retried the full agentic evaluator twice after the original rate-limit
  failure. The final attempt failed while the router requested structured
  output. All evaluator output remained outside `data/`.

### Exact failure output from the resumed retries

The first resumed agentic attempt failed with:

```text
groq.RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached
for model `qwen/qwen3.8-27b` ... input tokens per minute (ITPM): Limit 7000,
Used 6678, Requested 505. Please try again in 1.568571428s.',
'type': 'tokens', 'code': 'rate_limit_exceeded'}}
RuntimeError: Groq rate limit reached while processing the structured query
request. Please retry later.
```

The second, paced attempt failed with:

```text
groq.RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached
for model `qwen/qwen3.8-27b` ... output tokens per minute (OTPM): Limit 1000,
Used 721, Requested 763. Please try again in 29.04s.',
'type': 'tokens', 'code': 'rate_limit_exceeded'}}
RuntimeError: Groq rate limit reached while processing the structured query
request. Please retry later.
```

The third attempt, after setting the structured router's output cap to 200,
failed with:

```text
groq.BadRequestError: Error code: 400 - {'error': {'message': "Failed to call a
function. Please adjust your prompt. See 'failed_generation' for more details.",
'type': 'invalid_request_error', 'code': 'tool_use_failed',
'failed_generation': '\\u2028\\n\\n...'}}
```

### Suggested next step

Investigate Qwen's structured tool-call failure with the 200-token cap. Verify
whether `ChatGroq.with_structured_output(RouterIntent, method="json_mode")`
works with the existing JSON prompt, or adjust the output cap to allow valid
tool arguments while respecting the Groq quota. Then rerun the full Phase 8
agentic evaluation and compare overall success and category metrics against
`baseline_results/agentic_results.json` before proceeding to Phase 9.
