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


## Observability

### Logging

The kernel logs through the standard library under the `agent_kernel.engine` logger
(a child of `agent_kernel`), with structured fields attached to every record:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

| Event | Level | Extra fields |
| --- | --- | --- |
| decision made | `ALLOW` → INFO, `ESCALATE` → WARNING, `BLOCK` → ERROR | `request_id`, `verdict`, `tool_name`, `selected_tool`, `risk_score`, `confidence`, `latency_ms`, `model_alias`, `reason` |
| request retry | WARNING | `request_id`, `attempt`, `max_retries`, `status_code` |
| malformed / non-JSON judge response | ERROR | `request_id`, `tool_name` |
| HTTP failure | ERROR | `request_id`, `status_code` |

`decide()` issues exactly one decision record per call; `evaluate()`-level events
(request start at DEBUG, retries, parse/HTTP failures) carry the same `request_id`,
so a single request can be traced end to end.

### Audit trail

Every `decide()` call produces one `AuditEntry`:

```python
from agent_kernel import AuditEntry, JevSupervisorEngine

def persist(entry: AuditEntry) -> None:
    db.insert(entry.model_dump(mode="json"))

engine = JevSupervisorEngine(on_decision=persist)   # synchronous hook
engine = JevSupervisorEngine(audit_history_limit=0) # hook only, no history

decision = await engine.decide(action)
assert engine.audit[-1].decision == decision         # in-memory history
engine.clear_audit()
```

- `engine.audit` — tuple of entries, oldest first, bounded by `audit_history_limit`
  (default 1000; `0` disables history).
- `AuditEntry` — `request_id`, UTC `timestamp`, `model_alias`, `latency_ms`, the full
  `action`, and the final `decision`.
- `on_decision` is called synchronously after every decision. If it raises, the
  exception is logged and the decision is still returned — a broken audit sink can
  never block or alter governance.

## Escalation and guarded execution

### Escalation workflow

`ESCALATE` verdicts can be routed to a reviewer — a human queue, an approval ticket,
another model — by passing an `escalation_handler`. It receives an `EscalationRequest`
(`request_id`, `raised_at`, `action`, and the `ESCALATE` `decision`) and returns
`ALLOW`, `ESCALATE` (keep waiting), or `BLOCK`. It may be sync or async.

```python
from agent_kernel import JevSupervisorEngine

async def review(request):
    return "ALLOW" if await human_approved(request.request_id) else "BLOCK"

engine = JevSupervisorEngine(escalation_handler=review)
decision = await engine.decide(action)
```

- A handler that **raises** or returns an invalid verdict leaves the verdict at
  `ESCALATE` (fail-safe) and logs the failure.
- The resolution is recorded in `decision.reason` and the audit entry is marked
  `escalated=True`, with a dedicated WARNING log line for the escalation itself.

### Guarded execution

`execute()` is the agent-loop integration point: it decides first and only runs the
executor when the verdict is `ALLOW`.

```python
from agent_kernel import ActionBlocked

try:
    decision, result = await engine.execute(action, run_tool)   # sync or async executor
except ActionBlocked as blocked:
    handle(blocked.decision)  # .verdict, .reason, .risk_score, .confidence
```

`ActionBlocked` is raised for both `BLOCK` and `ESCALATE` (unless an escalation
handler approved it), so unreviewed work can never reach the executor.

## Command line

Installing the package provides an `agent-kernel` command (also runnable as
`python -m agent_kernel`):

```bash
agent-kernel run_bash \
  --args '{"cmd": "rm -rf build/"}' \
  --rationale "clean stale artifacts" \
  --task "rebuild the wheel" \
  --history "read_file pyproject.toml"
```

The decision is printed to stdout as JSON; the exit code is scriptable:

| Exit code | Meaning |
| --- | --- |
| 0 | `ALLOW` |
| 1 | error (bad input, network/judge failure) |
| 2 | usage error |
| 3 | `ESCALATE` |
| 4 | `BLOCK` |

```bash
if agent-kernel write_file --args '{"path": "/etc/hosts"}' > decision.json; then
  run_the_tool
else
  cat decision.json
fi
```

Add `--interactive` to resolve `ESCALATE` verdicts at the prompt (allow / block /
hold). Connection options: `--api-base-url`, `--model`, `--api-key` (or
`AGENT_KERNEL_API_KEY`), `--timeout`, `--retries`.

## Development

The project targets Python >= 3.11 and is developed inside WSL Debian.

```bash
cd ~/agent_kernel
python3 -m venv .venv
.venv/bin/pip install -e . pytest pytest-asyncio pytest-cov
.venv/bin/python -m pytest
```

A plain `pytest` run is the full gate: **all tests must pass and coverage of
`agent_kernel` must stay >= 90%** (`addopts` in `pyproject.toml` wires
`--cov=agent_kernel --cov-fail-under=90`). Use `pytest --no-cov` when iterating on a
single test, since a partial run cannot meet the coverage gate.

> If the repo lives on a WSL mount and you run pytest from Windows, set
> `COVERAGE_FILE` to a local directory first — sqlite (used by coverage) cannot lock
> reliably over `\\wsl.localhost`.

Test layout:

- `tests/test_engine.py` — policy gate (threshold boundaries, custom thresholds,
  unauthorized/mismatch rules), models (range and registry validation), prompt
  construction, response parsing, fail-closed `decide()`, HTTP retries/auth/pass-through,
  client lifecycle, and observability (audit entries, hooks, log levels and fields).
- `tests/test_http_integration.py` — end-to-end runs against a local judge server over
  a real socket: retries, bearer auth, request body, malformed bodies, HTTP errors, and
  connection failures.
- `tests/helpers.py` — shared builders plus `ScriptedJudge`, an `httpx.MockTransport`
  handler that replays scripted `(status, body)` responses and records every request.
  Unit tests never touch a socket; integration tests loop back to localhost — the suite
  never hits a live endpoint.
