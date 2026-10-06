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
| `blast_radius >= 3.5` | `ESCALATE` |
| `valid_tool confidence < 0.75` | `ESCALATE` |
| `0.40 <= is_destructive < 0.80` (uncertain zone) | `ESCALATE` |
| `is_destructive < 0.40` and `confidence >= 0.80` | `ALLOW` |
| anything else | `ESCALATE` (catch-all) |

Every verdict carries a human-readable `reason` string explaining exactly which rule fired.

## Usage

```python
import asyncio

from agent_kernel import JevSupervisorEngine, ProposedAction


async def main() -> None:
    action = ProposedAction(
        tool_name="run_bash",
        arguments={"cmd": "ls -la"},
        rationale="List workspace files before editing",
        current_task="Refactor the parser",
        execution_history=["read_file README.md"],
    )

    async with JevSupervisorEngine(api_base_url="https://api.typesafe.ai/v1",
                                   model_alias="jev-latest") as engine:
        decision = await engine.decide(action)
        print(decision["verdict"], decision["reason"])


asyncio.run(main())
```

`evaluate(action)` returns the raw `JevParallelResponse` if you want the judge's
reasoning; `decide(action)` additionally applies the policy gate.

## Development

The project targets Python >= 3.11 and is developed inside WSL Debian.

```bash
cd ~/agent_kernel
python3 -m venv .venv
.venv/bin/pip install -e . pytest pytest-asyncio pytest-cov
.venv/bin/python -m pytest -q
```

Tests live in `tests/test_engine.py` and cover the policy gate, the models, prompt
construction, response parsing, and HTTP failure handling. Network calls are always
mocked — the suite never hits a live endpoint.
