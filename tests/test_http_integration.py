import httpx
import pytest

from agent_kernel import DecisionResult, JevSupervisorEngine, ResponseParseError
from tests.helpers import make_action, payload


class TestEndToEndOverRealSocket:
    @pytest.mark.asyncio
    async def test_retry_auth_and_body_over_real_socket(self, judge_server):
        url, judge = judge_server
        judge.script.append((500, {"error": "transient"}))
        engine = JevSupervisorEngine(
            api_base_url=url,
            api_key="sk-e2e",
            max_retries=1,
            backoff_base=0,
            extra_body={"response_format": {"type": "json_object"}},
        )
        response = await engine.evaluate(make_action())
        assert response.blast_radius.score == 1.0
        assert len(judge.requests) == 2
        for request, body in zip(judge.requests, judge.bodies, strict=True):
            assert request.url.path == "/models/jev-latest/chat/completions"
            assert request.headers["authorization"] == "Bearer sk-e2e"
            assert body["model"] == "jev-latest"
            assert body["response_format"] == {"type": "json_object"}
            assert "Evaluate the following proposed agent action" in body["messages"][0]["content"]
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_blocks_over_real_socket(self, judge_server):
        url, judge = judge_server
        judge.script.append((200, payload(conf=0.99, prob=0.95, score_value=5.0)))
        engine = JevSupervisorEngine(api_base_url=url, max_retries=0)
        decision = await engine.decide(make_action())
        assert isinstance(decision, DecisionResult)
        assert decision.verdict == "BLOCK"
        assert len(judge.requests) == 1
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_fails_closed_on_malformed_body(self, judge_server):
        url, judge = judge_server
        malformed = {"choices": [payload()["choices"][0], payload()["choices"][1]]}
        judge.script.append((200, malformed))
        engine = JevSupervisorEngine(api_base_url=url, max_retries=0)
        decision = await engine.decide(make_action())
        assert decision.verdict == "ESCALATE"
        assert "malformed" in decision.reason
        assert len(judge.requests) == 1
        await engine.close()

    @pytest.mark.asyncio
    async def test_non_json_body_raises_parse_error(self, judge_server):
        url, judge = judge_server
        judge.script.append((200, b"<html>gateway error</html>"))
        engine = JevSupervisorEngine(api_base_url=url, max_retries=0)
        with pytest.raises(ResponseParseError):
            await engine.evaluate(make_action())
        await engine.close()

    @pytest.mark.asyncio
    async def test_non_retryable_status_propagates(self, judge_server):
        url, judge = judge_server
        judge.script.append((401, {"error": "bad key"}))
        engine = JevSupervisorEngine(api_base_url=url, max_retries=3, backoff_base=0)
        with pytest.raises(httpx.HTTPStatusError):
            await engine.evaluate(make_action())
        assert len(judge.requests) == 1
        await engine.close()

    @pytest.mark.asyncio
    async def test_connection_refused_raises_transport_error(self):
        engine = JevSupervisorEngine(
            api_base_url="http://127.0.0.1:9", max_retries=1, backoff_base=0, timeout=1.0
        )
        with pytest.raises(httpx.TransportError):
            await engine.evaluate(make_action())
        await engine.close()
