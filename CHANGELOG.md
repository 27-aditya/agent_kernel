# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-07

First release: a supervisor kernel that gates agent tool calls through an
LLM judge plus a deterministic policy gate, with fail-closed defaults.

### Added

- **Constrained judge outputs** — pydantic models for the judge's
  `choice`/`noul`/`score` responses with range validation, a fixed
  `VALID_TOOLS` registry, and `ResponseParseError` for malformed bodies.
- **Policy gate** (`PolicyGater`, `PolicyThresholds`) — deterministic
  thresholds over judge output: BLOCK on destructive probability,
  ESCALATE on unauthorized tool, tool mismatch, high blast radius, low
  confidence, or an uncertain zone; fails closed to ESCALATE.
- **Engine** (`JevSupervisorEngine`) — `evaluate()` and `decide()` with
  fail-closed `decide()`, HTTP retries with backoff, bearer auth
  (`api_key`/`AGENT_KERNEL_API_KEY`), `extra_body` pass-through, and
  async context-manager client lifecycle.
- **Observability** — bounded audit history, `on_decision` hook
  (fail-safe), and structured logging on the `agent_kernel.engine` logger
  with `request_id` tracing and distinct levels per verdict.
- **Escalation workflow** — optional sync/async `escalation_handler`
  receiving an `EscalationRequest`; invalid or raising handlers keep
  ESCALATE, resolutions are audited (`escalated=True`) and logged.
- **Guarded execution** — `execute()` runs an executor only on ALLOW and
  raises `ActionBlocked` (carrying the decision) otherwise.
- **CLI** — `agent-kernel` / `python -m agent_kernel` with scriptable
  exit codes (0 ALLOW, 1 error, 2 usage, 3 ESCALATE, 4 BLOCK), JSON
  decision on stdout, and `--interactive` escalation prompts (prompt on
  stderr so stdout stays machine-readable).
- **Quality gates** — 110 tests (>= 90% coverage gate via pytest-cov),
  real-socket integration tests against a scripted local judge, ruff
  lint, strict mypy, and a `dev` extra bundling the toolchain.
- Packaging metadata, MIT `LICENSE`, and this changelog.
