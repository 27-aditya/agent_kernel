# agent_kernel

A supervisor kernel that sits in front of an agent's tool calls and decides whether each
proposed action may run. Every action is scored by an LLM judge (`jev` family models) on
three axes in a single round trip, then a deterministic policy gate converts those scores
into one of three verdicts:

| Verdict | Meaning |
| --- | --- |
| `ALLOW` | Action is safe to execute. |
| `ESCALATE` | Too risky or too uncertain — requires human review. |
| `BLOCK` | Action is dangerous and must not run. |

## How it works

```
ProposedAction ──► JevSupervisorEngine ──► JevParallelResponse ──► PolicyGater ──► verdict
                       │                        │
                       │  _build_prompt()       │  choice  → valid_tool   (which tool, confidence)
                       │  POST /chat/completions│  noul    → is_destructive (probability)
                       │  _parse_response()     │  score   → blast_radius  (1–5)
```

1. **`engine.py` — `JevSupervisorEngine`**: builds a prompt from the proposed action,
   calls the model once for three parallel judgments, and parses the structured response.
2. **`models.py`**: pydantic models for inputs (`ProposedAction`), judge outputs
   (`JevParallelResponse`), and the final `DecisionResult`.
3. **`policy.py` — `PolicyGater`**: deterministic, code-only gate. No LLM in the decision path.

### Policy thresholds

Evaluated top to bottom; first match wins.

| Rule | Verdict |
| --- | --- |
| `is_destructive >= 0.80` | `BLOCK` |
| judge selected `unauthorized_tool` | `ESCALATE` |
| judge's selected tool != the action's `tool_name` | `ESCALATE` |
| `blast_radius >= 3.5` | `ESCALATE` |
| `valid_tool confidence < 0.75` | `ESCALATE` |
| `0.40 <= is_destructive < 0.80` (uncertain zone) | `ESCALATE` |
| `is_destructive < 0.40` and `confidence >= 0.80` | `ALLOW` |
| anything else | `ESCALATE` (catch-all) |

Thresholds are configurable via `PolicyThresholds` and injectable through
`PolicyGater(thresholds=...)` or `JevSupervisorEngine(policy=...)`; defaults are unchanged.
Every verdict carries a human-readable `reason` string explaining exactly which rule fired.

## Usage

```python
import asyncio

from agent_kernel import DecisionResult, JevSupervisorEngine, ProposedAction


async def main() -> None:
    action = ProposedAction(
        tool_name="run_bash",
        arguments={"cmd": "ls -la"},
        rationale="List workspace files before editing",
        current_task="Refactor the parser",
        execution_history=["read_file README.md"],
    )

    async with JevSupervisorEngine(
        api_base_url="https://api.typesafe.ai/v1",
        model_alias="jev-latest",
        api_key="sk-...",  # or set AGENT_KERNEL_API_KEY
    ) as engine:
        decision: DecisionResult = await engine.decide(action)
        print(decision.verdict, decision.reason)


asyncio.run(main())
```

`evaluate(action)` returns the raw `JevParallelResponse` if you want the judge's
reasoning; `decide(action)` additionally applies the policy gate and returns a
`DecisionResult` with `verdict`, `selected_tool`, `risk_score`, `confidence`, and `reason`.

## Resilience

- **Retries**: 429/500/502/503/504 and transport errors are retried with exponential
  backoff (`max_retries`, `backoff_base` constructor args; `max_retries=0` disables).
- **Auth**: `api_key` argument, falling back to the `AGENT_KERNEL_API_KEY` environment
  variable, is sent as `Authorization: Bearer <key>`.
- **Pass-through**: `extra_body={...}` is merged into the request JSON for
  API-specific options such as a structured-output format.
- **Fail closed**: a malformed judge response (missing/duplicate choices, out-of-range
  scores, unknown tool, non-JSON body) raises `ResponseParseError` from `evaluate()`.
  `decide()` converts it into an `ESCALATE` verdict, so nothing is ever auto-approved
  on bad judge output.


## Development

The project targets Python >= 3.11 and is developed inside WSL Debian.

```bash
cd ~/agent_kernel
python3 -m venv .venv
.venv/bin/pip install -e . pytest pytest-asyncio pytest-cov
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q --cov=agent_kernel --cov-report=term-missing
```

Tests live in `tests/test_engine.py` and cover the policy gate (including threshold
boundaries and custom thresholds), the models (range and registry validation), prompt
construction, response parsing (malformed, duplicate, out-of-range), fail-closed
`decide()`, and HTTP behaviour (retries, backoff, auth headers, pass-through body).
Network calls are always mocked — the suite never hits a live endpoint.
