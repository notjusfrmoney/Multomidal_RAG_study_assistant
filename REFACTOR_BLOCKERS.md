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
