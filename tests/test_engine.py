import logging

import httpx
import pytest
from pydantic import ValidationError

from agent_kernel import (
    ActionBlocked,
    AuditEntry,
    DecisionResult,
    JevSupervisorEngine,
    PolicyGater,
    PolicyThresholds,
    ProposedAction,
    ResponseParseError,
    VALID_TOOLS,
)
from agent_kernel.models import JevChoiceOutput, JevNoulOutput, JevScoreOutput
from tests.helpers import choice, make_action, make_engine, make_response, noul, payload, score


# ── Policy tests ────────────────────────────────────────────────────────────

class TestPolicyGater:
    def test_block_on_high_destructive(self):
        verdict, reason = PolicyGater().evaluate(make_response(conf=0.95, prob=0.90, blast=5.0))
        assert verdict == "BLOCK"
        assert "BLOCKED" in reason

    def test_escalate_on_blast_radius(self):
        verdict, reason = PolicyGater().evaluate(
            make_response(selected="write_file", conf=0.90, prob=0.30, blast=4.0)
        )
        assert verdict == "ESCALATE"
        assert "blast_radius" in reason.lower()

    def test_escalate_on_low_tool_confidence(self):
        verdict, reason = PolicyGater().evaluate(make_response(conf=0.60, prob=0.20, blast=2.0))
        assert verdict == "ESCALATE"
        assert "valid_tool confidence" in reason.lower()

    def test_escalate_on_uncertain_zone(self):
        verdict, reason = PolicyGater().evaluate(make_response(conf=0.90, prob=0.50, blast=2.0))
        assert verdict == "ESCALATE"
        assert "uncertain zone" in reason.lower()

    def test_allow_safe_action(self):
        verdict, reason = PolicyGater().evaluate(make_response())
        assert verdict == "ALLOW"
        assert "ALLOWED" in reason

    def test_escalate_catch_all(self):
        verdict, reason = PolicyGater().evaluate(make_response(conf=0.75, prob=0.35, blast=3.0))
        assert verdict == "ESCALATE"
        assert "catch-all" in reason.lower()

    def test_unauthorized_tool_always_escalates(self):
        verdict, reason = PolicyGater().evaluate(
            make_response(selected="unauthorized_tool", conf=0.99, prob=0.05, blast=1.0),
            make_action(tool_name="web_search"),
        )
        assert verdict == "ESCALATE"
        assert "unauthorized_tool" in reason

    def test_tool_mismatch_escalates(self):
        verdict, reason = PolicyGater().evaluate(
            make_response(selected="read_file", conf=0.99, prob=0.05, blast=1.0),
            make_action(tool_name="run_bash"),
        )
        assert verdict == "ESCALATE"
        assert "read_file" in reason
        assert "run_bash" in reason

    def test_matching_tool_allows(self):
        verdict, _ = PolicyGater().evaluate(
            make_response(selected="run_bash", conf=0.95, prob=0.10, blast=2.0),
            make_action(tool_name="run_bash"),
        )
        assert verdict == "ALLOW"

    def test_block_threshold_is_inclusive(self):
        verdict, _ = PolicyGater().evaluate(make_response(conf=0.95, prob=0.80, blast=1.0))
        assert verdict == "BLOCK"

    def test_uncertain_floor_is_inclusive(self):
        verdict, _ = PolicyGater().evaluate(make_response(conf=0.95, prob=0.40, blast=1.0))
        assert verdict == "ESCALATE"

    def test_just_below_allow_confidence_catch_all(self):
        verdict, reason = PolicyGater().evaluate(make_response(conf=0.79, prob=0.10, blast=1.0))
        assert verdict == "ESCALATE"
        assert "catch-all" in reason.lower()

    def test_allow_at_confidence_boundary(self):
        verdict, _ = PolicyGater().evaluate(make_response(conf=0.80, prob=0.10, blast=1.0))
        assert verdict == "ALLOW"

    def test_custom_thresholds_override_defaults(self):
        gater = PolicyGater(PolicyThresholds(escalate_blast_radius=4.5))
        verdict, _ = gater.evaluate(make_response(conf=0.95, prob=0.10, blast=4.0))
        assert verdict == "ALLOW"

    def test_custom_block_threshold(self):
        gater = PolicyGater(PolicyThresholds(block_destructive_prob=0.50))
        verdict, _ = gater.evaluate(make_response(conf=0.95, prob=0.50, blast=1.0))
        assert verdict == "BLOCK"


# ── Models tests ────────────────────────────────────────────────────────────

class TestModels:
    def test_proposed_action_creation(self):
        action = ProposedAction(
            tool_name="read_file",
            arguments={"path": "/src/main.py"},
            rationale="Need to inspect the main module",
            current_task="Fix login bug",
            execution_history=["read_file /README.md"],
        )
        assert action.tool_name == "read_file"
        assert len(action.execution_history) == 1

    def test_decision_result_allowed(self):
        result = DecisionResult(
            verdict="ALLOW", selected_tool="read_file", risk_score=1.0,
            confidence=0.95, reason="Safe action",
        )
        assert result.verdict == "ALLOW"

    def test_decision_result_blocked(self):
        result = DecisionResult(
            verdict="BLOCK", selected_tool="run_bash", risk_score=5.0,
            confidence=0.99, reason="Dangerous",
        )
        assert result.verdict == "BLOCK"

    def test_decision_result_invalid_raises(self):
        with pytest.raises(ValidationError):
            DecisionResult(
                verdict="PERMIT", selected_tool="read_file", risk_score=1.0,
                confidence=0.95, reason="",
            )

    @pytest.mark.parametrize("tool", list(VALID_TOOLS))
    def test_all_registry_tools_accepted(self, tool):
        assert JevChoiceOutput(selected_option=tool, confidence=0.5, reasoning="").selected_option == tool

    def test_unknown_tool_rejected(self):
        with pytest.raises(ValidationError):
            JevChoiceOutput(selected_option="web_search", confidence=0.9, reasoning="")

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            JevChoiceOutput(selected_option="read_file", confidence=1.2, reasoning="")

    def test_negative_probability_rejected(self):
        with pytest.raises(ValidationError):
            JevNoulOutput(probability=-0.01, reasoning="")

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            JevScoreOutput(score=6.0, reasoning="")
        with pytest.raises(ValidationError):
            JevScoreOutput(score=0.5, reasoning="")


# ── Engine tests ────────────────────────────────────────────────────────────

class TestJevSupervisorEngine:
    @pytest.mark.asyncio
    async def test_evaluate_returns_valid_response(self):
        engine, judge = make_engine(max_retries=0)
        result = await engine.evaluate(make_action())
        assert result.valid_tool.selected_option == "read_file"
        assert result.is_destructive.probability == 0.10
        assert result.blast_radius.score == 1.0
        assert len(judge.requests) == 1
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_returns_decision_result(self):
        engine, _ = make_engine(max_retries=0)
        decision = await engine.decide(make_action())
        assert isinstance(decision, DecisionResult)
        assert decision.verdict == "ALLOW"
        assert decision.selected_tool == "read_file"
        assert decision.risk_score == 1.0
        assert decision.confidence == 0.95
        assert "ALLOWED" in decision.reason
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_blocks_destructive(self):
        engine, _ = make_engine(script=[payload("run_bash", 0.99, 0.95, 5.0)], max_retries=0)
        decision = await engine.decide(make_action(tool_name="run_bash"))
        assert decision.verdict == "BLOCK"
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_escalates_uncertain_zone(self):
        engine, _ = make_engine(script=[payload("write_file", 0.90, 0.60, 2.0)], max_retries=0)
        decision = await engine.decide(make_action(tool_name="write_file"))
        assert decision.verdict == "ESCALATE"
        assert "uncertain zone" in decision.reason.lower()
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_escalates_low_tool_confidence(self):
        engine, _ = make_engine(script=[payload("run_bash", 0.50, 0.20, 2.0)], max_retries=0)
        decision = await engine.decide(make_action(tool_name="run_bash"))
        assert decision.verdict == "ESCALATE"
        assert "valid_tool confidence" in decision.reason.lower()
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_escalates_high_blast_radius(self):
        engine, _ = make_engine(script=[payload("run_bash", 0.90, 0.30, 4.5)], max_retries=0)
        decision = await engine.decide(make_action(tool_name="run_bash"))
        assert decision.verdict == "ESCALATE"
        assert decision.risk_score == 4.5
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_escalates_unauthorized_tool(self):
        engine, _ = make_engine(script=[payload("unauthorized_tool", 0.99, 0.05, 1.0)], max_retries=0)
        decision = await engine.decide(make_action(tool_name="web_search"))
        assert decision.verdict == "ESCALATE"
        assert "unauthorized_tool" in decision.reason
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_escalates_tool_mismatch(self):
        engine, _ = make_engine(script=[payload("write_file", 0.99, 0.05, 1.0)], max_retries=0)
        decision = await engine.decide(make_action(tool_name="run_bash"))
        assert decision.verdict == "ESCALATE"
        assert "write_file" in decision.reason
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_uses_injected_policy(self):
        engine, _ = make_engine(
            script=[payload("run_bash", 0.95, 0.10, 4.0)],
            max_retries=0,
            policy=PolicyGater(PolicyThresholds(escalate_blast_radius=4.5)),
        )
        decision = await engine.decide(make_action(tool_name="run_bash"))
        assert decision.verdict == "ALLOW"
        await engine.close()

    @pytest.mark.asyncio
    async def test_build_prompt_contains_criteria(self):
        engine, _ = make_engine()
        prompt = engine._build_prompt(make_action(tool_name="run_bash"))
        assert "valid_tool" in prompt
        assert "unauthorized_tool" in prompt
        assert "is_destructive" in prompt
        assert "blast_radius" in prompt
        assert "1=purely read-only" in prompt
        for tool in VALID_TOOLS:
            assert tool in prompt
        await engine.close()

    @pytest.mark.asyncio
    async def test_raise_on_http_error(self):
        engine, judge = make_engine(script=[(500, {"error": "boom"})], max_retries=0)
        with pytest.raises(httpx.HTTPStatusError):
            await engine.evaluate(make_action())
        assert len(judge.requests) == 1
        await engine.close()

    @pytest.mark.asyncio
    async def test_context_manager_closes_client(self):
        engine, _ = make_engine()
        async with engine:
            client = engine._get_client()
            assert not client.is_closed
        assert client.is_closed

    @pytest.mark.asyncio
    async def test_client_is_reused_across_calls(self):
        engine, judge = make_engine(max_retries=0)
        first = engine._get_client()
        second = engine._get_client()
        assert first is second
        await engine.evaluate(make_action())
        assert len(judge.requests) == 1
        await engine.close()

    @pytest.mark.asyncio
    async def test_fresh_client_after_close(self):
        engine, _ = make_engine()
        first = engine._get_client()
        await engine.close()
        assert first.is_closed
        second = engine._get_client()
        assert second is not first
        assert not second.is_closed
        await engine.close()


# ── Response parsing tests ──────────────────────────────────────────────────

class TestResponseParsing:
    BAD_PAYLOADS = [
        pytest.param(
            {"choices": [choice(), noul()]},
            "exactly one 'score'",
            id="missing-score-choice",
        ),
        pytest.param(
            {"choices": [choice(), noul(), noul(), score()]},
            "exactly one 'noul'",
            id="duplicate-noul-choice",
        ),
        pytest.param(
            {"choices": [{"type": "choice", "content": "just text"}, noul(), score()]},
            "no 'content' object",
            id="content-not-an-object",
        ),
        pytest.param(
            {"choices": "not-a-list"},
            "missing the 'choices' array",
            id="choices-not-a-list",
        ),
        pytest.param(
            {"error": "rate limited"},
            "missing the 'choices' array",
            id="no-choices-key",
        ),
        pytest.param(
            {"choices": [choice("web_search"), noul(), score()]},
            "invalid values",
            id="unknown-tool-option",
        ),
        pytest.param(
            {"choices": [choice(conf=1.5), noul(), score()]},
            "invalid values",
            id="confidence-out-of-range",
        ),
        pytest.param(
            {"choices": [choice(), noul(), score(7.0)]},
            "invalid values",
            id="score-out-of-range",
        ),
        pytest.param(
            {"choices": [choice(), noul(prob=1.4), score()]},
            "invalid values",
            id="probability-out-of-range",
        ),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_payload,fragment", BAD_PAYLOADS)
    async def test_evaluate_raises_on_bad_payload(self, bad_payload, fragment):
        engine, _ = make_engine(script=[bad_payload], max_retries=0)
        with pytest.raises(ResponseParseError) as exc:
            await engine.evaluate(make_action())
        assert fragment in str(exc.value)
        await engine.close()

    @pytest.mark.asyncio
    async def test_evaluate_raises_on_non_json_body(self):
        engine, _ = make_engine(script=[b"<html>nope</html>"], max_retries=0)
        with pytest.raises(ResponseParseError) as exc:
            await engine.evaluate(make_action())
        assert "not valid JSON" in str(exc.value)
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_fails_closed_on_malformed_response(self):
        engine, _ = make_engine(script=[{"choices": [choice(), noul()]}], max_retries=0)
        decision = await engine.decide(make_action())
        assert isinstance(decision, DecisionResult)
        assert decision.verdict == "ESCALATE"
        assert "malformed" in decision.reason
        assert decision.confidence == 0.0
        await engine.close()


# ── HTTP resilience tests ───────────────────────────────────────────────────

class TestHttpResilience:
    @pytest.mark.asyncio
    async def test_retries_retryable_status_then_succeeds(self):
        engine, judge = make_engine(
            script=[(500, {"error": "transient"}), payload()], max_retries=2, backoff_base=0
        )
        result = await engine.evaluate(make_action())
        assert result.valid_tool.selected_option == "read_file"
        assert len(judge.requests) == 2
        await engine.close()

    @pytest.mark.asyncio
    async def test_retries_exhausted_raises_http_error(self):
        engine, judge = make_engine(
            script=[(503, {"error": "down"}), (503, {"error": "down"})], max_retries=1, backoff_base=0
        )
        with pytest.raises(httpx.HTTPStatusError):
            await engine.evaluate(make_action())
        assert len(judge.requests) == 2
        await engine.close()

    @pytest.mark.asyncio
    async def test_non_retryable_status_raises_immediately(self):
        engine, judge = make_engine(script=[(401, {"error": "bad key"})], max_retries=3, backoff_base=0)
        with pytest.raises(httpx.HTTPStatusError):
            await engine.evaluate(make_action())
        assert len(judge.requests) == 1
        await engine.close()

    @pytest.mark.asyncio
    async def test_retries_transport_error_then_succeeds(self):
        engine, judge = make_engine(
            script=[httpx.ConnectError, payload()], max_retries=2, backoff_base=0
        )
        result = await engine.evaluate(make_action())
        assert result.blast_radius.score == 1.0
        assert len(judge.requests) == 2
        await engine.close()

    @pytest.mark.asyncio
    async def test_transport_error_exhausted_raises(self):
        engine, judge = make_engine(
            script=[httpx.ConnectError, httpx.ConnectError], max_retries=1, backoff_base=0
        )
        with pytest.raises(httpx.ConnectError):
            await engine.evaluate(make_action())
        assert len(judge.requests) == 2
        await engine.close()

    @pytest.mark.asyncio
    async def test_api_key_sent_and_extra_body_merged(self, monkeypatch):
        monkeypatch.delenv("AGENT_KERNEL_API_KEY", raising=False)
        engine, judge = make_engine(max_retries=0, api_key="sk-test",
                                    extra_body={"response_format": {"type": "json_object"}})
        await engine.evaluate(make_action())
        request = judge.requests[0]
        body = judge.bodies[0]
        assert request.headers["authorization"] == "Bearer sk-test"
        assert body["model"] == "jev-latest"
        assert body["response_format"] == {"type": "json_object"}
        assert "content" in body["messages"][0]
        await engine.close()

    @pytest.mark.asyncio
    async def test_no_auth_header_without_key(self, monkeypatch):
        monkeypatch.delenv("AGENT_KERNEL_API_KEY", raising=False)
        engine, judge = make_engine(max_retries=0)
        await engine.evaluate(make_action())
        assert "authorization" not in judge.requests[0].headers
        await engine.close()

    @pytest.mark.asyncio
    async def test_api_key_from_environment(self, monkeypatch):
        monkeypatch.setenv("AGENT_KERNEL_API_KEY", "sk-env")
        engine = JevSupervisorEngine(api_base_url="http://judge.test")
        assert engine.api_key == "sk-env"
        assert engine._headers() == {"Authorization": "Bearer sk-env"}
        await engine.close()


# ── Observability tests ─────────────────────────────────────────────────────

class TestObservability:
    @pytest.mark.asyncio
    async def test_audit_entry_recorded(self):
        engine, _ = make_engine(max_retries=0)
        decision = await engine.decide(make_action())
        entries = engine.audit
        assert len(entries) == 1
        entry = entries[0]
        assert isinstance(entry, AuditEntry)
        assert entry.decision == decision
        assert entry.request_id
        assert entry.model_alias == "jev-latest"
        assert entry.latency_ms >= 0
        assert entry.action.tool_name == "read_file"
        assert entry.timestamp.tzinfo is not None
        await engine.close()

    @pytest.mark.asyncio
    async def test_on_decision_hook_receives_every_entry(self):
        collected: list[AuditEntry] = []
        engine, _ = make_engine(max_retries=0, on_decision=collected.append)
        for _ in range(2):
            await engine.decide(make_action())
        assert collected == list(engine.audit)
        assert len(collected) == 2
        await engine.close()

    @pytest.mark.asyncio
    async def test_hook_failure_does_not_break_decision(self, caplog):
        def boom(entry: AuditEntry) -> None:
            raise RuntimeError("hook exploded")

        engine, _ = make_engine(max_retries=0, on_decision=boom)
        with caplog.at_level(logging.ERROR, logger="agent_kernel.engine"):
            decision = await engine.decide(make_action())
        assert decision.verdict == "ALLOW"
        assert any("on_decision hook raised" in r.getMessage() for r in caplog.records)
        await engine.close()

    @pytest.mark.asyncio
    async def test_audit_history_is_bounded(self):
        engine, _ = make_engine(max_retries=0, audit_history_limit=2)
        request_ids = []
        for _ in range(3):
            await engine.decide(make_action())
            request_ids.append(engine.audit[-1].request_id)
        assert len(engine.audit) == 2
        assert [e.request_id for e in engine.audit] == request_ids[1:]
        await engine.close()

    @pytest.mark.asyncio
    async def test_audit_history_can_be_disabled_but_hook_still_fires(self):
        collected: list[AuditEntry] = []
        engine, _ = make_engine(max_retries=0, audit_history_limit=0,
                                on_decision=collected.append)
        await engine.decide(make_action())
        assert engine.audit == ()
        assert len(collected) == 1
        await engine.close()

    @pytest.mark.asyncio
    async def test_clear_audit(self):
        engine, _ = make_engine(max_retries=0)
        await engine.decide(make_action())
        assert len(engine.audit) == 1
        engine.clear_audit()
        assert engine.audit == ()
        await engine.close()

    @pytest.mark.asyncio
    async def test_request_ids_are_unique_per_decision(self):
        engine, _ = make_engine(max_retries=0)
        await engine.decide(make_action())
        await engine.decide(make_action())
        ids = [e.request_id for e in engine.audit]
        assert len(set(ids)) == 2
        await engine.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "prob,conf,blast,expected_verdict,expected_level,expected_risk",
        [
            pytest.param(0.10, 0.95, 1.0, "ALLOW", logging.INFO, 1.0, id="allow"),
            pytest.param(0.60, 0.95, 1.0, "ESCALATE", logging.WARNING, 1.0, id="escalate"),
            pytest.param(0.95, 0.99, 5.0, "BLOCK", logging.ERROR, 5.0, id="block"),
        ],
    )
    async def test_decision_log_level_and_structured_fields(
        self, caplog, prob, conf, blast, expected_verdict, expected_level, expected_risk
    ):
        engine, _ = make_engine(script=[payload("read_file", conf, prob, blast)], max_retries=0)
        with caplog.at_level(logging.DEBUG, logger="agent_kernel.engine"):
            decision = await engine.decide(make_action())
        assert decision.verdict == expected_verdict
        record = next(
            r for r in caplog.records
            if getattr(r, "verdict", None) == expected_verdict
            and getattr(r, "tool_name", None) == "read_file"
        )
        assert record.levelno == expected_level
        assert record.request_id
        assert record.selected_tool == "read_file"
        assert record.tool_name == "read_file"
        assert record.risk_score == expected_risk
        assert record.confidence == conf
        assert record.latency_ms >= 0
        assert record.model_alias == "jev-latest"
        assert record.reason
        await engine.close()

    @pytest.mark.asyncio
    async def test_retry_logged_as_warning_with_context(self, caplog):
        engine, _ = make_engine(script=[(500, {"error": "transient"}), payload()],
                                max_retries=1, backoff_base=0)
        with caplog.at_level(logging.WARNING, logger="agent_kernel.engine"):
            await engine.evaluate(make_action())
        retry = [r for r in caplog.records if getattr(r, "attempt", None) == 1]
        assert retry, "expected a retry log record"
        assert retry[0].levelno == logging.WARNING
        assert retry[0].status_code == 500
        assert retry[0].request_id
        await engine.close()

    @pytest.mark.asyncio
    async def test_malformed_response_logs_error_and_still_escalates(self, caplog):
        engine, _ = make_engine(script=[{"choices": [choice(), noul()]}], max_retries=0)
        with caplog.at_level(logging.ERROR, logger="agent_kernel.engine"):
            decision = await engine.decide(make_action())
        assert decision.verdict == "ESCALATE"
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any("malformed response" in r.getMessage() for r in errors)
        assert len(engine.audit) == 1
        assert engine.audit[0].decision.verdict == "ESCALATE"
        await engine.close()

    @pytest.mark.asyncio
    async def test_http_error_logged_with_status(self, caplog):
        engine, _ = make_engine(script=[(500, {"error": "boom"})], max_retries=0)
        with caplog.at_level(logging.ERROR, logger="agent_kernel.engine"):
            with pytest.raises(httpx.HTTPStatusError):
                await engine.evaluate(make_action())
        errors = [r for r in caplog.records if getattr(r, "status_code", None) == 500]
        assert errors and errors[0].levelno == logging.ERROR
        await engine.close()


# ── Escalation workflow tests ───────────────────────────────────────────────

ESCALATE_PAYLOAD = payload("read_file", 0.95, 0.60, 1.0)  # uncertain zone


class TestEscalationWorkflow:
    @pytest.mark.asyncio
    async def test_handler_resolves_escalate_to_allow(self):
        engine, _ = make_engine(script=[ESCALATE_PAYLOAD], max_retries=0,
                                escalation_handler=lambda req: "ALLOW")
        decision = await engine.decide(make_action())
        assert decision.verdict == "ALLOW"
        assert "ESCALATE" in decision.reason
        assert "Escalation resolved to ALLOW" in decision.reason
        assert engine.audit[0].escalated is True
        await engine.close()

    @pytest.mark.asyncio
    async def test_handler_resolves_escalate_to_block(self):
        engine, _ = make_engine(script=[ESCALATE_PAYLOAD], max_retries=0,
                                escalation_handler=lambda req: "BLOCK")
        decision = await engine.decide(make_action())
        assert decision.verdict == "BLOCK"
        assert "Escalation resolved to BLOCK" in decision.reason
        await engine.close()

    @pytest.mark.asyncio
    async def test_handler_can_keep_escalate(self):
        engine, _ = make_engine(script=[ESCALATE_PAYLOAD], max_retries=0,
                                escalation_handler=lambda req: "ESCALATE")
        decision = await engine.decide(make_action())
        assert decision.verdict == "ESCALATE"
        assert "Escalation resolved" not in decision.reason
        assert engine.audit[0].escalated is True
        await engine.close()

    @pytest.mark.asyncio
    async def test_async_handler_supported(self):
        seen = {}

        async def async_handler(req):
            seen["request_id"] = req.request_id
            seen["action"] = req.action
            seen["verdict"] = req.decision.verdict
            return "ALLOW"

        engine, _ = make_engine(script=[ESCALATE_PAYLOAD], max_retries=0,
                                escalation_handler=async_handler)
        decision = await engine.decide(make_action())
        assert decision.verdict == "ALLOW"
        assert seen["verdict"] == "ESCALATE"
        assert seen["action"].tool_name == "read_file"
        assert seen["request_id"] == engine.audit[0].request_id
        await engine.close()

    @pytest.mark.asyncio
    async def test_invalid_handler_verdict_keeps_escalate(self, caplog):
        engine, _ = make_engine(script=[ESCALATE_PAYLOAD], max_retries=0,
                                escalation_handler=lambda req: "PERMIT")
        with caplog.at_level(logging.ERROR, logger="agent_kernel.engine"):
            decision = await engine.decide(make_action())
        assert decision.verdict == "ESCALATE"
        assert any("escalation handler returned 'PERMIT'" in r.getMessage()
                   for r in caplog.records)
        await engine.close()

    @pytest.mark.asyncio
    async def test_handler_exception_keeps_escalate(self, caplog):
        def broken(req):
            raise RuntimeError("review queue down")

        engine, _ = make_engine(script=[ESCALATE_PAYLOAD], max_retries=0,
                                escalation_handler=broken)
        with caplog.at_level(logging.ERROR, logger="agent_kernel.engine"):
            decision = await engine.decide(make_action())
        assert decision.verdict == "ESCALATE"
        assert any("keeping ESCALATE" in r.getMessage() for r in caplog.records)
        assert len(engine.audit) == 1
        await engine.close()

    @pytest.mark.asyncio
    async def test_handler_not_called_when_allowed(self):
        calls = []
        engine, _ = make_engine(max_retries=0, escalation_handler=lambda req: calls.append(req))
        decision = await engine.decide(make_action())
        assert decision.verdict == "ALLOW"
        assert calls == []
        assert engine.audit[0].escalated is False
        await engine.close()

    @pytest.mark.asyncio
    async def test_escalation_logged_and_audited_once(self, caplog):
        engine, _ = make_engine(script=[ESCALATE_PAYLOAD], max_retries=0,
                                escalation_handler=lambda req: "ALLOW")
        with caplog.at_level(logging.DEBUG, logger="agent_kernel.engine"):
            await engine.decide(make_action())
        escalations = [r for r in caplog.records if "awaiting escalation handler" in r.getMessage()]
        assert len(escalations) == 1
        assert escalations[0].request_id
        decisions = [r for r in caplog.records if getattr(r, "verdict", None) == "ALLOW"]
        assert decisions and decisions[0].escalated is True
        assert len(engine.audit) == 1
        await engine.close()

    @pytest.mark.asyncio
    async def test_no_handler_leaves_escalate(self):
        engine, _ = make_engine(script=[ESCALATE_PAYLOAD], max_retries=0)
        decision = await engine.decide(make_action())
        assert decision.verdict == "ESCALATE"
        assert engine.audit[0].escalated is False
        await engine.close()


# ── Executor integration tests ──────────────────────────────────────────────

class TestExecute:
    @pytest.mark.asyncio
    async def test_execute_runs_executor_on_allow(self):
        engine, _ = make_engine(max_retries=0)
        calls = []

        def executor(action):
            calls.append(action.tool_name)
            return "file contents"

        decision, result = await engine.execute(make_action(), executor)
        assert decision.verdict == "ALLOW"
        assert result == "file contents"
        assert calls == ["read_file"]
        await engine.close()

    @pytest.mark.asyncio
    async def test_execute_supports_async_executor(self):
        engine, _ = make_engine(max_retries=0)

        async def executor(action):
            return {"status": "ok"}

        _, result = await engine.execute(make_action(), executor)
        assert result == {"status": "ok"}
        await engine.close()

    @pytest.mark.asyncio
    async def test_execute_raises_on_block(self):
        engine, _ = make_engine(script=[payload("run_bash", 0.99, 0.95, 5.0)], max_retries=0)
        calls = []

        def executor(action):
            calls.append(action)
            return "should not run"

        with pytest.raises(ActionBlocked) as exc:
            await engine.execute(make_action(tool_name="run_bash"), executor)
        assert exc.value.decision.verdict == "BLOCK"
        assert calls == []
        await engine.close()

    @pytest.mark.asyncio
    async def test_execute_raises_on_escalate(self):
        engine, _ = make_engine(script=[ESCALATE_PAYLOAD], max_retries=0)
        calls = []

        def executor(action):
            calls.append(action)

        with pytest.raises(ActionBlocked) as exc:
            await engine.execute(make_action(), executor)
        assert exc.value.decision.verdict == "ESCALATE"
        assert calls == []
        await engine.close()

    @pytest.mark.asyncio
    async def test_execute_runs_after_escalation_resolution(self):
        engine, _ = make_engine(script=[ESCALATE_PAYLOAD], max_retries=0,
                                escalation_handler=lambda req: "ALLOW")

        def executor(action):
            return f"ran {action.tool_name}"

        decision, result = await engine.execute(make_action(), executor)
        assert decision.verdict == "ALLOW"
        assert decision.reason  # escalation provenance preserved
        assert result == "ran read_file"
        assert engine.audit[0].escalated is True
        await engine.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
