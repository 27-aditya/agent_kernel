import httpx
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pydantic import ValidationError

from agent_kernel.models import ProposedAction, DecisionResult, JevParallelResponse, JevChoiceOutput, JevNoulOutput, JevScoreOutput
from agent_kernel.policy import PolicyGater
from agent_kernel.engine import JevSupervisorEngine


# ── Policy tests ─────────────────────────────────────────────────────────────

class TestPolicyGater:
    def test_block_on_high_destructive(self):
        gater = PolicyGater()
        resp = JevParallelResponse(
            valid_tool=JevChoiceOutput(selected_option="run_bash", confidence=0.95, reasoning=""),
            is_destructive=JevNoulOutput(probability=0.90, reasoning=""),
            blast_radius=JevScoreOutput(score=5.0, reasoning=""),
        )
        verdict, reason = gater.evaluate(resp)
        assert verdict == "BLOCK"
        assert "BLOCKED" in reason

    def test_escalate_on_blast_radius(self):
        gater = PolicyGater()
        resp = JevParallelResponse(
            valid_tool=JevChoiceOutput(selected_option="write_file", confidence=0.90, reasoning=""),
            is_destructive=JevNoulOutput(probability=0.30, reasoning=""),
            blast_radius=JevScoreOutput(score=4.0, reasoning=""),
        )
        verdict, reason = gater.evaluate(resp)
        assert verdict == "ESCALATE"
        assert "blast_radius" in reason.lower()

    def test_escalate_on_low_tool_confidence(self):
        gater = PolicyGater()
        resp = JevParallelResponse(
            valid_tool=JevChoiceOutput(selected_option="run_bash", confidence=0.60, reasoning=""),
            is_destructive=JevNoulOutput(probability=0.20, reasoning=""),
            blast_radius=JevScoreOutput(score=2.0, reasoning=""),
        )
        verdict, reason = gater.evaluate(resp)
        assert verdict == "ESCALATE"
        assert "valid_tool confidence" in reason.lower()

    def test_escalate_on_uncertain_zone(self):
        gater = PolicyGater()
        resp = JevParallelResponse(
            valid_tool=JevChoiceOutput(selected_option="read_file", confidence=0.90, reasoning=""),
            is_destructive=JevNoulOutput(probability=0.50, reasoning=""),
            blast_radius=JevScoreOutput(score=2.0, reasoning=""),
        )
        verdict, reason = gater.evaluate(resp)
        assert verdict == "ESCALATE"
        assert "uncertain zone" in reason.lower()

    def test_allow_safe_action(self):
        gater = PolicyGater()
        resp = JevParallelResponse(
            valid_tool=JevChoiceOutput(selected_option="read_file", confidence=0.95, reasoning=""),
            is_destructive=JevNoulOutput(probability=0.10, reasoning=""),
            blast_radius=JevScoreOutput(score=1.0, reasoning=""),
        )
        verdict, reason = gater.evaluate(resp)
        assert verdict == "ALLOW"
        assert "ALLOWED" in reason

    def test_escalate_catch_all(self):
        gater = PolicyGater()
        resp = JevParallelResponse(
            valid_tool=JevChoiceOutput(selected_option="run_bash", confidence=0.75, reasoning=""),
            is_destructive=JevNoulOutput(probability=0.35, reasoning=""),
            blast_radius=JevScoreOutput(score=3.0, reasoning=""),
        )
        verdict, reason = gater.evaluate(resp)
        assert verdict == "ESCALATE"
        assert "catch-all" in reason.lower()


# ── Models tests ─────────────────────────────────────────────────────────────

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
            verdict="ALLOW",
            selected_tool="read_file",
            risk_score=1.0,
            confidence=0.95,
            reason="Safe action",
        )
        assert result.verdict == "ALLOW"

    def test_decision_result_blocked(self):
        result = DecisionResult(
            verdict="BLOCK",
            selected_tool="run_bash",
            risk_score=5.0,
            confidence=0.99,
            reason="Dangerous",
        )
        assert result.verdict == "BLOCK"

    def test_decision_result_invalid_raises(self):
        with pytest.raises(ValidationError):
            DecisionResult(
                verdict="PERMIT",
                selected_tool="read_file",
                risk_score=1.0,
                confidence=0.95,
                reason="",
            )


# ── Engine tests ─────────────────────────────────────────────────────────────

class TestJevSupervisorEngine:
    @pytest.mark.asyncio
    async def test_evaluate_returns_valid_response(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost")
        action = ProposedAction(
            tool_name="read_file",
            arguments={"path": "/test"},
            rationale="testing",
            current_task="test",
            execution_history=[],
        )
        mock_response = {
            "choices": [
                {"type": "choice", "content": {"selected_option": "read_file", "confidence": 0.95, "reasoning": "Read-only tool"}},
                {"type": "noul", "content": {"probability": 0.10, "reasoning": "No risk"}},
                {"type": "score", "content": {"score": 1.0, "reasoning": "Read-only"}},
            ]
        }
        with patch.object(engine._get_client(), "post", new=AsyncMock(return_value=MagicMock(status_code=200, json=lambda: mock_response))) as mock_post:  # type: ignore[union-attr]
            result = await engine.evaluate(action)
            assert result.valid_tool.selected_option == "read_file"
            assert result.is_destructive.probability == 0.10
            assert result.blast_radius.score == 1.0
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_allows_safe_action(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost")
        action = ProposedAction(
            tool_name="read_file",
            arguments={"path": "/test"},
            rationale="testing",
            current_task="test",
            execution_history=[],
        )
        mock_response = {
            "choices": [
                {"type": "choice", "content": {"selected_option": "read_file", "confidence": 0.95, "reasoning": ""}},
                {"type": "noul", "content": {"probability": 0.10, "reasoning": ""}},
                {"type": "score", "content": {"score": 1.0, "reasoning": ""}},
            ]
        }
        with patch.object(engine._get_client(), "post", new=AsyncMock(return_value=MagicMock(status_code=200, json=lambda: mock_response))):  # type: ignore[union-attr]
            decision = await engine.decide(action)
            assert decision["verdict"] == "ALLOW"
            assert decision["risk_score"] == 1.0
            assert decision["confidence"] == 0.95
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_blocks_destructive(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost")
        action = ProposedAction(
            tool_name="run_bash",
            arguments={"cmd": "rm -rf /"},
            rationale="clean up",
            current_task="test",
            execution_history=[],
        )
        mock_response = {
            "choices": [
                {"type": "choice", "content": {"selected_option": "run_bash", "confidence": 0.99, "reasoning": ""}},
                {"type": "noul", "content": {"probability": 0.95, "reasoning": ""}},
                {"type": "score", "content": {"score": 5.0, "reasoning": ""}},
            ]
        }
        with patch.object(engine._get_client(), "post", new=AsyncMock(return_value=MagicMock(status_code=200, json=lambda: mock_response))):  # type: ignore[union-attr]
            decision = await engine.decide(action)
            assert decision["verdict"] == "BLOCK"
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_escalates_uncertain_zone(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost")
        action = ProposedAction(
            tool_name="write_file",
            arguments={"path": "/etc/config"},
            rationale="update config",
            current_task="test",
            execution_history=[],
        )
        mock_response = {
            "choices": [
                {"type": "choice", "content": {"selected_option": "write_file", "confidence": 0.90, "reasoning": ""}},
                {"type": "noul", "content": {"probability": 0.60, "reasoning": ""}},
                {"type": "score", "content": {"score": 2.0, "reasoning": ""}},
            ]
        }
        with patch.object(engine._get_client(), "post", new=AsyncMock(return_value=MagicMock(status_code=200, json=lambda: mock_response))):  # type: ignore[union-attr]
            decision = await engine.decide(action)
            assert decision["verdict"] == "ESCALATE"
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_escalates_low_tool_confidence(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost")
        action = ProposedAction(
            tool_name="unauthorized_tool",
            arguments={},
            rationale="testing",
            current_task="test",
            execution_history=[],
        )
        mock_response = {
            "choices": [
                {"type": "choice", "content": {"selected_option": "unauthorized_tool", "confidence": 0.50, "reasoning": ""}},
                {"type": "noul", "content": {"probability": 0.20, "reasoning": ""}},
                {"type": "score", "content": {"score": 2.0, "reasoning": ""}},
            ]
        }
        with patch.object(engine._get_client(), "post", new=AsyncMock(return_value=MagicMock(status_code=200, json=lambda: mock_response))):  # type: ignore[union-attr]
            decision = await engine.decide(action)
            assert decision["verdict"] == "ESCALATE"
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_escalates_high_blast_radius(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost")
        action = ProposedAction(
            tool_name="run_bash",
            arguments={"cmd": "systemctl restart"},
            rationale="restart service",
            current_task="test",
            execution_history=[],
        )
        mock_response = {
            "choices": [
                {"type": "choice", "content": {"selected_option": "run_bash", "confidence": 0.90, "reasoning": ""}},
                {"type": "noul", "content": {"probability": 0.30, "reasoning": ""}},
                {"type": "score", "content": {"score": 4.5, "reasoning": ""}},
            ]
        }
        with patch.object(engine._get_client(), "post", new=AsyncMock(return_value=MagicMock(status_code=200, json=lambda: mock_response))):  # type: ignore[union-attr]
            decision = await engine.decide(action)
            assert decision["verdict"] == "ESCALATE"
            assert decision["risk_score"] == 4.5
        await engine.close()

    @pytest.mark.asyncio
    async def test_build_prompt_contains_criteria(self):
        engine = JevSupervisorEngine()
        action = ProposedAction(
            tool_name="run_bash",
            arguments={},
            rationale="",
            current_task="",
            execution_history=[],
        )
        prompt = engine._build_prompt(action)
        assert "valid_tool" in prompt
        assert "unauthorized_tool" in prompt
        assert "is_destructive" in prompt
        assert "blast_radius" in prompt
        assert "1=purely read-only" in prompt

    @pytest.mark.asyncio
    async def test_raise_on_http_error(self):
        engine = JevSupervisorEngine(api_base_url="http://localhost")
        action = ProposedAction(
            tool_name="read_file",
            arguments={},
            rationale="",
            current_task="",
            execution_history=[],
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("Error", request=MagicMock(), response=mock_resp)
        with patch.object(engine._get_client(), "post", new=AsyncMock(return_value=mock_resp)):  # type: ignore[union-attr]
            with pytest.raises(httpx.HTTPStatusError):
                await engine.evaluate(action)
        await engine.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])