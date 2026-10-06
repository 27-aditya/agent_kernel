import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from agent_kernel import DecisionResult, JevSupervisorEngine, ProposedAction, ResponseParseError

DEFAULT_PAYLOAD = {
    "choices": [
        {"type": "choice",
         "content": {"selected_option": "run_bash", "confidence": 0.95, "reasoning": "ok"}},
        {"type": "noul", "content": {"probability": 0.10, "reasoning": "ok"}},
        {"type": "score", "content": {"score": 2.0, "reasoning": "ok"}},
    ]
}


class _State:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.script: list[tuple[int, object]] = []


class _JudgeHandler(BaseHTTPRequestHandler):
    state: _State

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        self.state.requests.append(
            {
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": json.loads(raw) if raw else None,
            }
        )
        status, payload = self.state.script.pop(0) if self.state.script else (200, DEFAULT_PAYLOAD)
        if isinstance(payload, bytes):
            data, content_type = payload, "text/html"
        else:
            data, content_type = json.dumps(payload).encode(), "application/json"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # silence request logging
        pass


@pytest.fixture
def judge_server():
    state = _State()
    handler = type("BoundHandler", (_JudgeHandler,), {"state": state})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", state
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def make_action(tool_name: str = "run_bash") -> ProposedAction:
    return ProposedAction(
        tool_name=tool_name,
        arguments={"cmd": "ls -la"},
        rationale="list files",
        current_task="inspect workspace",
        execution_history=[],
    )


def payload(selected: str = "run_bash", conf: float = 0.95, prob: float = 0.10,
            score_value: float = 2.0) -> dict:
    return {
        "choices": [
            {"type": "choice",
             "content": {"selected_option": selected, "confidence": conf, "reasoning": "ok"}},
            {"type": "noul", "content": {"probability": prob, "reasoning": "ok"}},
            {"type": "score", "content": {"score": score_value, "reasoning": "ok"}},
        ]
    }


class TestEndToEndOverRealSocket:
    @pytest.mark.asyncio
    async def test_retry_auth_and_body_over_real_socket(self, judge_server):
        url, state = judge_server
        state.script.append((500, {"error": "transient"}))
        engine = JevSupervisorEngine(
            api_base_url=url,
            api_key="sk-e2e",
            max_retries=1,
            backoff_base=0,
            extra_body={"response_format": {"type": "json_object"}},
        )
        response = await engine.evaluate(make_action())
        assert response.blast_radius.score == 2.0
        assert len(state.requests) == 2
        for req in state.requests:
            assert req["path"] == "/models/jev-latest/chat/completions"
            assert req["headers"]["authorization"] == "Bearer sk-e2e"
            assert req["body"]["model"] == "jev-latest"
            assert req["body"]["response_format"] == {"type": "json_object"}
            assert "Evaluate the following proposed agent action" in req["body"]["messages"][0]["content"]
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_blocks_over_real_socket(self, judge_server):
        url, state = judge_server
        state.script.append((200, payload(conf=0.99, prob=0.95, score_value=5.0)))
        engine = JevSupervisorEngine(api_base_url=url, max_retries=0)
        decision = await engine.decide(make_action())
        assert isinstance(decision, DecisionResult)
        assert decision.verdict == "BLOCK"
        assert len(state.requests) == 1
        await engine.close()

    @pytest.mark.asyncio
    async def test_decide_fails_closed_on_malformed_body(self, judge_server):
        url, state = judge_server
        malformed = {"choices": [payload()["choices"][0], payload()["choices"][1]]}
        state.script.append((200, malformed))
        engine = JevSupervisorEngine(api_base_url=url, max_retries=0)
        decision = await engine.decide(make_action())
        assert decision.verdict == "ESCALATE"
        assert "malformed" in decision.reason
        assert len(state.requests) == 1
        await engine.close()

    @pytest.mark.asyncio
    async def test_non_json_body_raises_parse_error(self, judge_server):
        url, state = judge_server
        state.script.append((200, b"<html>gateway error</html>"))
        engine = JevSupervisorEngine(api_base_url=url, max_retries=0)
        with pytest.raises(ResponseParseError):
            await engine.evaluate(make_action())
        await engine.close()

    @pytest.mark.asyncio
    async def test_non_retryable_status_propagates(self, judge_server):
        url, state = judge_server
        state.script.append((401, {"error": "bad key"}))
        engine = JevSupervisorEngine(api_base_url=url, max_retries=3, backoff_base=0)
        with pytest.raises(httpx.HTTPStatusError):
            await engine.evaluate(make_action())
        assert len(state.requests) == 1
        await engine.close()

    @pytest.mark.asyncio
    async def test_connection_refused_raises_transport_error(self):
        engine = JevSupervisorEngine(
            api_base_url="http://127.0.0.1:9", max_retries=1, backoff_base=0, timeout=1.0
        )
        with pytest.raises(httpx.TransportError):
            await engine.evaluate(make_action())
        await engine.close()
