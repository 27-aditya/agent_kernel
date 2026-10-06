# agent_kernel

`agent_kernel` is a supervisor kernel for agent tool calls.

The kernel sits in front of the tools that an agent uses. For each proposed action,
it makes one decision. The decision is `ALLOW`, `ESCALATE`, or `BLOCK`.

The decision comes from two stages:

1. An LLM judge scores the action on three axes in one call.
2. A policy gate turns the scores into a verdict.

The policy gate is deterministic code. It does not use an LLM.

| Verdict  | Meaning |
| --- | --- |
| `ALLOW`    | The action is safe to execute. |
| `ESCALATE` | The action is too risky or too uncertain. It needs human review. |
| `BLOCK`    | The action is dangerous. It must not run. |

## Terms

| Term       | Meaning |
| ---        | --- |
| Action     | The tool call that the agent wants to make. |
| Judge      | The LLM model that scores the action. |
| Verdict    | The final result: `ALLOW`, `ESCALATE`, or `BLOCK`. |
| Policy gate| The code that checks the judge scores against fixed rules. |
| Escalation | The route of an `ESCALATE` verdict to a reviewer. |

## How it works

```text
ProposedAction ──► JevSupervisorEngine ──► JevParallelResponse ──► PolicyGater ──► verdict
                      │                        │
                      │  build prompt           choice ──► valid_tool
                      │  POST /chat/completions noul ──► is_destructive
                      │  parse response         score ──► blast_radius
```

The flow is the same for every call:

1. The engine builds a prompt from the proposed action.
2. The engine calls the judge model once.
3. The judge returns one structured response.
4. The response contains three judgments:
   - a `choice` object for `valid_tool` (which tool to use, and confidence)
   - a `noul` object for `is_destructive` (probability of damage)
   - a `score` object for `blast_radius` (a score from 1 to 5)
5. The policy gate applies its rules to these values.
6. The policy gate returns the verdict.

Source layout:

- `agent_kernel/engine.py` — `JevSupervisorEngine`. It builds the prompt, makes the
  call, and parses the response.
- `agent_kernel/models.py` — the pydantic models for inputs, outputs, and decisions.
- `agent_kernel/policy.py` — `PolicyGater`. It applies the deterministic rules.
- `agent_kernel/cli.py` — the `agent-kernel` command.

## Install

The project requires Python 3.11 or newer. The package is not on PyPI.

Install it from a checked-out copy of the repository:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"   # development install
.venv/bin/pip install .             # regular install
```

Or install a built wheel:

```bash
.venv/bin/pip install dist/*.whl
```

## Quick start

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

`evaluate(action)` returns the raw `JevParallelResponse`. Use it when you want the
judge's scores. `decide(action)` also applies the policy gate. It returns a
`DecisionResult`. `execute(action, executor)` runs `decide` first, then runs the
executor only for `ALLOW`. See [Escalation and guarded execution](#escalation-and-guarded-execution).

### Models

`ProposedAction` describes the tool call. It has five fields.

| Field              | Type           | Description |
| ---                | ---            | --- |
| `tool_name`        | `str`          | The name of the tool to call. |
| `arguments`        | `dict[str, Any]` | The arguments for the tool. |
| `rationale`        | `str`          | Why the agent wants to run the tool. |
| `current_task`     | `str`          | The task that the agent works on. |
| `execution_history`| `list[str]`    | The actions that the agent ran before. |

`DecisionResult` is the final result. It has five fields.

| Field           | Type    | Description |
| ---             | ---     | --- |
| `verdict`       | `Verdict` | `ALLOW`, `ESCALATE`, or `BLOCK`. |
| `selected_tool` | `str`   | The tool that the judge selected. |
| `risk_score`    | `float` | The blast radius score (1 to 5). |
| `confidence`    | `float` | The judge confidence (0 to 1). |
| `reason`        | `str`   | Why the policy gate chose this verdict. |

The `reason` string always states which rule fired. Use it to trace the decision.

## Policy rules

The policy gate checks the rules from top to bottom. The first rule that matches
wins.

| Rule | Verdict |
| --- | --- |
| `is_destructive >= 0.80` | `BLOCK` |
| the judge selected `unauthorized_tool` | `ESCALATE` |
| the selected tool differs from the action `tool_name` | `ESCALATE` |
| `blast_radius >= 3.5` | `ESCALATE` |
| `valid_tool confidence < 0.75` | `ESCALATE` |
| `0.40 <= is_destructive < 0.80` (the uncertain zone) | `ESCALATE` |
| `is_destructive < 0.40` and `confidence >= 0.80` | `ALLOW` |
| any other case | `ESCALATE` (catch-all) |

Use `PolicyThresholds` to change the defaults. Pass the thresholds to
`PolicyGater(thresholds=...)`. To inject a custom gate, pass
`policy=...` to `JevSupervisorEngine`.

| Threshold | Default |
| --- | --- |
| `block_destructive_prob` | 0.80 |
| `escalate_blast_radius` | 3.5 |
| `min_tool_confidence` | 0.75 |
| `uncertain_prob_floor` | 0.40 |
| `allow_min_confidence` | 0.80 |

## Judge protocol

The judge answers three questions in one response:

- `Choice` — "valid_tool". The options are the tool names in `VALID_TOOLS`.
- `Noul` — "is_destructive". The answer is a probability (0 to 1).
- `Score` — "blast_radius". The answer is a score (1 to 5).

`VALID_TOOLS` contains six names.

- `run_bash`
- `read_file`
- `write_file`
- `git_checkout`
- `git_commit`
- `unauthorized_tool`

A judge response must contain exactly one `choice`, exactly one `noul`, and exactly
one `score` object. Every value must stay inside its range. The chosen tool must be
in `VALID_TOOLS`.

If the response does not follow these rules, the engine treats it as a failure:

- `evaluate()` raises `ResponseParseError`.
- `decide()` returns `ESCALATE`.

This is the fail-closed behavior. Bad judge output never becomes `ALLOW`.

## Resilience

The engine handles judge failures in a controlled way.

- **Retries**: The engine retries on 429, 500, 502, 503, 504, and transport errors.
  It uses exponential backoff. Set `max_retries` and `backoff_base` in the
  constructor. Set `max_retries=0` to disable retries.
- **Auth**: The engine reads the key from the `api_key` argument. If the argument is
  absent, it reads `AGENT_KERNEL_API_KEY` from the environment. The key is sent as
  the `Authorization: Bearer <key>` header.
- **Pass-through**: `extra_body={...}` merges into the request JSON. Use it for
  API-specific options, such as a structured-output format.
- **Fail closed**: Any malformed judge response keeps the verdict at `ESCALATE`.
  See [Judge protocol](#judge-protocol).

## Observability

### Logging

The kernel logs to the standard library logger `agent_kernel.engine`. Every record
carries structured fields.

```python
import logging

logging.basicConfig(level=logging.INFO)
```

| Event | Level | Extra fields |
| --- | --- | --- |
| decision made | `ALLOW` → INFO, `ESCALATE` → WARNING, `BLOCK` → ERROR | `request_id`, `verdict`, `tool_name`, `selected_tool`, `risk_score`, `confidence`, `latency_ms`, `model_alias`, `reason` |
| request retry | WARNING | `request_id`, `attempt`, `max_retries`, `status_code` |
| malformed or non-JSON judge response | ERROR | `request_id`, `tool_name` |
| HTTP failure | ERROR | `request_id`, `status_code` |

`decide()` writes exactly one decision record per call. All earlier events share the
same `request_id`. Use the `request_id` to trace one request from start to end.

### Audit trail

Every `decide()` call writes one `AuditEntry`.

```python
from agent_kernel import AuditEntry, JevSupervisorEngine

def persist(entry: AuditEntry) -> None:
    db.insert(entry.model_dump(mode="json"))

engine = JevSupervisorEngine(on_decision=persist)   # a synchronous hook
```

- `engine.audit` is a tuple of entries. It is bounded by `audit_history_limit`
  (default 1000). Set `audit_history_limit=0` to disable history and use the hook only.
- `AuditEntry` contains `request_id`, a UTC `timestamp`, `model_alias`, `latency_ms`,
  the full `action`, the final `decision`, and the `escalated` flag.
- The `on_decision` hook runs synchronously after every decision. If the hook raises,
  the engine logs the exception and returns the decision anyway. A broken audit sink
  never blocks governance.
- `engine.clear_audit()` removes all history.

## Escalation and guarded execution

### Escalation workflow

An `ESCALATE` verdict keeps the action from running. You can route it to a reviewer
with an `escalation_handler`. The reviewer can be a human queue, an approval ticket,
or another model.

Pass the handler to the engine constructor. The handler receives an
`EscalationRequest` with `request_id`, `raised_at`, `action`, and the `ESCALATE`
`decision`. The handler returns `ALLOW`, `ESCALATE` (keep waiting), or `BLOCK`.
It may be sync or async.

```python
from agent_kernel import JevSupervisorEngine

async def review(request):
    return "ALLOW" if await human_approved(request.request_id) else "BLOCK"

engine = JevSupervisorEngine(escalation_handler=review)
decision = await engine.decide(action)
```

Rules for the handler:

- If the handler raises, the verdict stays `ESCALATE`. The engine logs the failure.
- If the handler returns an invalid verdict, the verdict stays `ESCALATE`.
- If the handler resolves to `ALLOW` or `BLOCK`, the engine copies the decision with
  the new verdict. It appends the resolution to `decision.reason`.
- The audit entry is marked `escalated=True`.
- The escalation itself writes a WARNING log line.

### Guarded execution

`execute()` is the integration point for an agent loop. It decides first, then runs
the executor only when the verdict is `ALLOW`.

```python
from agent_kernel import ActionBlocked

try:
    decision, result = await engine.execute(action, run_tool)   # sync or async executor
except ActionBlocked as blocked:
    handle(blocked.decision)  # .verdict, .reason, .risk_score, .confidence
```

`ActionBlocked` is raised for both `BLOCK` and `ESCALATE`. It carries the decision
in the `decision` attribute. Unreviewed work never reaches the executor.

## Command line

The package provides the `agent-kernel` command. You can also run
`python -m agent_kernel`.

```bash
agent-kernel run_bash \
  --args '{"cmd": "rm -rf build/"}' \
  --rationale "clean stale artifacts" \
  --task "rebuild the wheel" \
  --history "read_file pyproject.toml"
```

The command prints the decision to stdout as JSON. The exit code is scriptable.

| Exit code | Meaning |
| --- | --- |
| 0 | `ALLOW` |
| 1 | error (bad input, network, or judge failure) |
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

Output rules:

- stdout carries only the JSON decision. This keeps the output machine-readable.
- All messages and prompts go to stderr.
- `--interactive` resolves `ESCALATE` verdicts at a prompt. Type `y` for allow,
  `b` for block, or `a` for hold.

Connection options: `--api-base-url`, `--model`, `--api-key` (or
`AGENT_KERNEL_API_KEY`), `--timeout`, and `--retries`.

## Development

The project targets Python 3.11 or newer. Development takes place in WSL Debian.

```bash
cd ~/agent_kernel
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest     # the full gate: tests and coverage
.venv/bin/ruff check .         # lint (ruff config in pyproject.toml)
.venv/bin/mypy                 # typecheck (strict, package only)
```

A plain `pytest` run is the full gate. All tests must pass. Coverage of
`agent_kernel` must stay at or above 90%. The `addopts` in `pyproject.toml` wires
`--cov=agent_kernel --cov-fail-under=90`. Use `pytest --no-cov` while you work on a
single test, because a partial run cannot meet the coverage gate.

> Note for Windows users: if the repository lives on a WSL mount, set
> `COVERAGE_FILE` to a local directory first. The sqlite database that coverage uses
> cannot lock reliably over `\\wsl.localhost`.

Test layout:

- `tests/test_engine.py` — the policy gate, the models, the prompt, response parsing,
  fail-closed `decide()`, retries, auth, pass-through, client lifecycle, logging and
  audit, the escalation workflow, and guarded `execute()`.
- `tests/test_cli.py` — CLI exit codes, JSON output, argument validation, and
  interactive escalation, including `python -m agent_kernel` subprocesses.
- `tests/test_http_integration.py` — end-to-end runs against a local judge server
  over a real socket. Covers retries, bearer auth, the request body, malformed
  bodies, HTTP errors, and connection failures.
- `tests/conftest.py` — the shared `judge_server` fixture. It is a local scripted
  judge over a real socket.
- `tests/helpers.py` — shared builders and `ScriptedJudge`, an `httpx.MockTransport`
  handler that replays scripted `(status, body)` responses and records every request.

Unit tests never touch a socket. Integration tests loop back to localhost. The suite
never hits a live endpoint.

GitHub Actions (`.github/workflows/ci.yml`) runs the same gates on every push and
pull request. It runs on Python 3.11, 3.12, and 3.13. The checks are ruff, strict
mypy, and the pytest coverage gate. The workflow also builds the sdist and the wheel
and runs `twine check`.

Validate a release locally:

```bash
.venv/bin/pip install build twine
.venv/bin/python -m build
.venv/bin/twine check dist/*
```

## License

[MIT](LICENSE)