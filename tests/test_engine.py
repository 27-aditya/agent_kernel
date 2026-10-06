import json
import logging

import httpx
import pytest
from pydantic import ValidationError
from unittest.mock import AsyncMock, patch, MagicMock

from agent_kernel import (
    AuditEntry,
    DecisionResult,
    JevSupervisorEngine,
    PolicyGater,
    PolicyThresholds,
    ProposedAction,
    ResponseParseError,
    VALID_TOOLS,
)
from agent_kernel.models import JevChoiceOutput, JevNoulOutput, JevParallelResponse, JevScoreOutput


# ── Builders ────────────────────────────────────────────────────────────────

def choice(selected: str = "read_file", conf: float = 0.95) -> dict:
    return {
        "type": "choice",
        "content": {"selected_option": selected, "confidence": conf, "reasoning": "r"},
    }


def noul(prob: float = 0.10) -> dict:
    return {"type": "noul", "content": {"probability": prob, "reasoning": "r"}}


def score(value: float = 1.0) -> dict:
    return {"type": "score", "content": {"score": value, "reasoning": "r"}}


def default_payload() -> dict:
    return {"choices": [choice(), noul(), score()]}


def make_action(tool_name: str = "read_file") -> ProposedAction:
    return ProposedAction(
        tool_name=tool_name,
        arguments={"path": "/test"},
        rationale="testing",
        current_task="test",
        execution_history=[],
    )


def make_response(
    selected: str = "read_file",
    conf: float = 0.95,
    prob: float = 0.10,
    blast: float = 1.0,
) -> JevParallelResponse:
    return JevParallelResponse(
        valid_tool=JevChoiceOutput(selected_option=selected, confidence=conf, reasoning=""),
        is_destructive=JevNoulOutput(probability=prob, reasoning=""),
        blast_radius=JevScoreOutput(score=blast, reasoning=""),
    )


def mock_http(payload: dict | None = None, status_code: int = 200,
              json_error: Exception | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status.return_value = None
    if json_error is not None:
        resp.json.side_effect = json_error
    else:
        resp.json.return_value = payload if payload is not None else default_payload()
    return resp


def http_error(resp: MagicMock) -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError("server error", request=MagicMock(), response=resp)


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
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        with patch.object(engine._get_client(), "post",
                          new=AsyncMock(return_value=mock_http())) as post:
            result = await engine.evaluate(make_action())
            assert result.valid_tool.selected_option == "read_file"
            assert result.is_destructive.probability == 0.10
            assert result.blast_radius.score == 1.0
            assert post.await_count == 1
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_returns_decision_result(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        with patch.object(engine._get_client(), "post",
                          new=AsyncMock(return_value=mock_http())):
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
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        payload = {"choices": [choice("run_bash", 0.99), noul(0.95), score(5.0)]}
        with patch.object(engine._get_client(), "post",
                          new=AsyncMock(return_value=mock_http(payload))):
            decision = await engine.decide(make_action(tool_name="run_bash"))
            assert decision.verdict == "BLOCK"
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_escalates_uncertain_zone(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        payload = {"choices": [choice("write_file", 0.90), noul(0.60), score(2.0)]}
        with patch.object(engine._get_client(), "post",
                          new=AsyncMock(return_value=mock_http(payload))):
            decision = await engine.decide(make_action(tool_name="write_file"))
            assert decision.verdict == "ESCALATE"
            assert "uncertain zone" in decision.reason.lower()
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_escalates_low_tool_confidence(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        payload = {"choices": [choice("run_bash", 0.50), noul(0.20), score(2.0)]}
        with patch.object(engine._get_client(), "post",
                          new=AsyncMock(return_value=mock_http(payload))):
            decision = await engine.decide(make_action(tool_name="run_bash"))
            assert decision.verdict == "ESCALATE"
            assert "valid_tool confidence" in decision.reason.lower()
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_escalates_high_blast_radius(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        payload = {"choices": [choice("run_bash", 0.90), noul(0.30), score(4.5)]}
        with patch.object(engine._get_client(), "post",
                          new=AsyncMock(return_value=mock_http(payload))):
            decision = await engine.decide(make_action(tool_name="run_bash"))
            assert decision.verdict == "ESCALATE"
            assert decision.risk_score == 4.5
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_escalates_unauthorized_tool(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        payload = {"choices": [choice("unauthorized_tool", 0.99), noul(0.05), score(1.0)]}
        with patch.object(engine._get_client(), "post",
                          new=AsyncMock(return_value=mock_http(payload))):
            decision = await engine.decide(make_action(tool_name="web_search"))
            assert decision.verdict == "ESCALATE"
            assert "unauthorized_tool" in decision.reason
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_escalates_tool_mismatch(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        payload = {"choices": [choice("write_file", 0.99), noul(0.05), score(1.0)]}
        with patch.object(engine._get_client(), "post",
                          new=AsyncMock(return_value=mock_http(payload))):
            decision = await engine.decide(make_action(tool_name="run_bash"))
            assert decision.verdict == "ESCALATE"
            assert "write_file" in decision.reason
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_uses_injected_policy(self):
        engine = JevSupervisorEngine(
            api_base_url="http://localhost",
            max_retries=0,
            policy=PolicyGater(PolicyThresholds(escalate_blast_radius=4.5)),
        )
        payload = {"choices": [choice("run_bash", 0.95), noul(0.10), score(4.0)]}
        with patch.object(engine._get_client(), "post",
                          new=AsyncMock(return_value=mock_http(payload))):
            decision = await engine.decide(make_action(tool_name="run_bash"))
            assert decision.verdict == "ALLOW"
        await engine.close()

    @pytest.mark.asyncio
    async def test_build_prompt_contains_criteria(self):
        engine = JevSupervisorEngine()
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
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        bad = mock_http(status_code=500)
        bad.raise_for_status.side_effect = http_error(bad)
        with patch.object(engine._get_client(), "post", new=AsyncMock(return_value=bad)):
            with pytest.raises(httpx.HTTPStatusError):
                await engine.evaluate(make_action())
        await engine.close()

    @pytest.mark.asyncio
    async def test_context_manager_closes_client(self):
        async with JevSupervisorEngine(api_base_url="http://localhost") as engine:
            client = engine._get_client()
            assert not client.is_closed
        assert client.is_closed


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
    @pytest.mark.parametrize("payload,fragment", BAD_PAYLOADS)
    async def test_evaluate_raises_on_bad_payload(self, payload, fragment):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        with patch.object(engine._get_client(), "post",
                          new=AsyncMock(return_value=mock_http(payload))):
            with pytest.raises(ResponseParseError) as exc:
                await engine.evaluate(make_action())
            assert fragment in str(exc.value)
        await engine.close()

    @pytest.mark.asyncio
    async def test_evaluate_raises_on_non_json_body(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        resp = mock_http(json_error=ValueError("no json here"))
        with patch.object(engine._get_client(), "post", new=AsyncMock(return_value=resp)):
            with pytest.raises(ResponseParseError) as exc:
                await engine.evaluate(make_action())
            assert "not valid JSON" in str(exc.value)
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_fails_closed_on_malformed_response(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        payload = {"choices": [choice(), noul()]}
        with patch.object(engine._get_client(), "post",
                          new=AsyncMock(return_value=mock_http(payload))):
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
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=2, backoff_base=0)
        server_error = mock_http(status_code=500)
        with patch.object(engine._get_client(), "post",
                          new=AsyncMock(side_effect=[server_error, mock_http()])) as post:
            result = await engine.evaluate(make_action())
            assert result.valid_tool.selected_option == "read_file"
            assert post.await_count == 2
        await engine.close()

    @pytest.mark.asyncio
    async def test_retries_exhausted_raises_http_error(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=1, backoff_base=0)
        bad = mock_http(status_code=503)
        bad.raise_for_status.side_effect = http_error(bad)
        with patch.object(engine._get_client(), "post",
                          new=AsyncMock(side_effect=[bad, bad])) as post:
            with pytest.raises(httpx.HTTPStatusError):
                await engine.evaluate(make_action())
            assert post.await_count == 2
        await engine.close()

    @pytest.mark.asyncio
    async def test_non_retryable_status_raises_immediately(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=3, backoff_base=0)
        bad = mock_http(status_code=401)
        bad.raise_for_status.side_effect = http_error(bad)
        with patch.object(engine._get_client(), "post",
                          new=AsyncMock(return_value=bad)) as post:
            with pytest.raises(httpx.HTTPStatusError):
                await engine.evaluate(make_action())
            assert post.await_count == 1
        await engine.close()

    @pytest.mark.asyncio
    async def test_retries_transport_error_then_succeeds(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=2, backoff_base=0)
        request = httpx.Request("POST", "http://localhost/v1/chat")
        connect_error = httpx.ConnectError("connection refused", request=request)
        with patch.object(engine._get_client(), "post",
                          new=AsyncMock(side_effect=[connect_error, mock_http()])) as post:
            result = await engine.evaluate(make_action())
            assert result.blast_radius.score == 1.0
            assert post.await_count == 2
        await engine.close()

    @pytest.mark.asyncio
    async def test_transport_error_exhausted_raises(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=1, backoff_base=0)
        request = httpx.Request("POST", "http://localhost/v1/chat")
        connect_error = httpx.ConnectError("connection refused", request=request)
        with patch.object(engine._get_client(), "post",
                          new=AsyncMock(side_effect=[connect_error, connect_error])) as post:
            with pytest.raises(httpx.ConnectError):
                await engine.evaluate(make_action())
            assert post.await_count == 2
        await engine.close()

    @pytest.mark.asyncio
    async def test_api_key_sent_and_extra_body_merged(self, monkeypatch):
        monkeypatch.delenv("AGENT_KERNEL_API_KEY", raising=False)
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["authorization"] = request.headers.get("authorization")
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=default_payload())

        engine = JevSupervisorEngine(
            api_base_url="http://localhost",
            api_key="sk-test",
            extra_body={"response_format": {"type": "json_object"}},
        )
        engine._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await engine.evaluate(make_action())
        assert seen["authorization"] == "Bearer sk-test"
        assert seen["body"]["model"] == "jev-latest"
        assert seen["body"]["response_format"] == {"type": "json_object"}
        assert "content" in seen["body"]["messages"][0]
        await engine.close()

    @pytest.mark.asyncio
    async def test_no_auth_header_without_key(self, monkeypatch):
        monkeypatch.delenv("AGENT_KERNEL_API_KEY", raising=False)
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["authorization"] = request.headers.get("authorization")
            return httpx.Response(200, json=default_payload())

        engine = JevSupervisorEngine(api_base_url="http://localhost")
        engine._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await engine.evaluate(make_action())
        assert seen["authorization"] is None
        await engine.close()

    @pytest.mark.asyncio
    async def test_api_key_from_environment(self, monkeypatch):
        monkeypatch.setenv("AGENT_KERNEL_API_KEY", "sk-env")
        engine = JevSupervisorEngine(api_base_url="http://localhost")
        assert engine.api_key == "sk-env"
        assert engine._headers() == {"Authorization": "Bearer sk-env"}
        await engine.close()


# ── Observability tests ─────────────────────────────────────────────────────

class TestObservability:
    @pytest.mark.asyncio
    async def test_audit_entry_recorded(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        with patch.object(engine._get_client(), "post", new=AsyncMock(return_value=mock_http())):
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
        engine = JevSupervisorEngine(
            api_base_url="http://localhost", max_retries=0, on_decision=collected.append
        )
        for _ in range(2):
            with patch.object(engine._get_client(), "post",
                              new=AsyncMock(return_value=mock_http())):
                await engine.decide(make_action())
        assert collected == list(engine.audit)
        assert len(collected) == 2
        await engine.close()

    @pytest.mark.asyncio
    async def test_hook_failure_does_not_break_decision(self, caplog):
        def boom(entry: AuditEntry) -> None:
            raise RuntimeError("hook exploded")

        engine = JevSupervisorEngine(
            api_base_url="http://localhost", max_retries=0, on_decision=boom
        )
        with caplog.at_level(logging.ERROR, logger="agent_kernel.engine"):
            with patch.object(engine._get_client(), "post",
                              new=AsyncMock(return_value=mock_http())):
                decision = await engine.decide(make_action())
        assert decision.verdict == "ALLOW"
        assert any("on_decision hook raised" in r.getMessage() for r in caplog.records)
        await engine.close()

    @pytest.mark.asyncio
    async def test_audit_history_is_bounded(self):
        engine = JevSupervisorEngine(
            api_base_url="http://localhost", max_retries=0, audit_history_limit=2
        )
        request_ids = []
        for _ in range(3):
            with patch.object(engine._get_client(), "post",
                              new=AsyncMock(return_value=mock_http())):
                await engine.decide(make_action())
            request_ids.append(engine.audit[-1].request_id)
        assert len(engine.audit) == 2
        assert [e.request_id for e in engine.audit] == request_ids[1:]
        await engine.close()

    @pytest.mark.asyncio
    async def test_audit_history_can_be_disabled_but_hook_still_fires(self):
        collected: list[AuditEntry] = []
        engine = JevSupervisorEngine(
            api_base_url="http://localhost", max_retries=0,
            audit_history_limit=0, on_decision=collected.append,
        )
        with patch.object(engine._get_client(), "post", new=AsyncMock(return_value=mock_http())):
            await engine.decide(make_action())
        assert engine.audit == ()
        assert len(collected) == 1
        await engine.close()

    @pytest.mark.asyncio
    async def test_clear_audit(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        with patch.object(engine._get_client(), "post", new=AsyncMock(return_value=mock_http())):
            await engine.decide(make_action())
        assert len(engine.audit) == 1
        engine.clear_audit()
        assert engine.audit == ()
        await engine.close()

    @pytest.mark.asyncio
    async def test_request_ids_are_unique_per_decision(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        with patch.object(engine._get_client(), "post", new=AsyncMock(return_value=mock_http())):
            first = await engine.decide(make_action())
            second = await engine.decide(make_action())
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
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        payload = {"choices": [choice("read_file", conf), noul(prob), score(blast)]}
        with caplog.at_level(logging.DEBUG, logger="agent_kernel.engine"):
            with patch.object(engine._get_client(), "post",
                              new=AsyncMock(return_value=mock_http(payload))):
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
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=1, backoff_base=0)
        with caplog.at_level(logging.WARNING, logger="agent_kernel.engine"):
            with patch.object(engine._get_client(), "post",
                              new=AsyncMock(side_effect=[mock_http(status_code=500), mock_http()])):
                await engine.evaluate(make_action())
        retry = [r for r in caplog.records if getattr(r, "attempt", None) == 1]
        assert retry, "expected a retry log record"
        assert retry[0].levelno == logging.WARNING
        assert retry[0].status_code == 500
        assert retry[0].request_id
        await engine.close()

    @pytest.mark.asyncio
    async def test_malformed_response_logs_error_and_still_escalates(self, caplog):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        payload = {"choices": [choice(), noul()]}
        with caplog.at_level(logging.ERROR, logger="agent_kernel.engine"):
            with patch.object(engine._get_client(), "post",
                              new=AsyncMock(return_value=mock_http(payload))):
                decision = await engine.decide(make_action())
        assert decision.verdict == "ESCALATE"
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any("malformed response" in r.getMessage() for r in errors)
        assert len(engine.audit) == 1
        assert engine.audit[0].decision.verdict == "ESCALATE"
        await engine.close()

    @pytest.mark.asyncio
    async def test_http_error_logged_with_status(self, caplog):
        engine = JevSupervisorEngine(api_base_url="http://localhost", max_retries=0)
        bad = mock_http(status_code=500)
        bad.raise_for_status.side_effect = http_error(bad)
        with caplog.at_level(logging.ERROR, logger="agent_kernel.engine"):
            with patch.object(engine._get_client(), "post", new=AsyncMock(return_value=bad)):
                with pytest.raises(httpx.HTTPStatusError):
                    await engine.evaluate(make_action())
        errors = [r for r in caplog.records if getattr(r, "status_code", None) == 500]
        assert errors and errors[0].levelno == logging.ERROR
        await engine.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
