"""Shared builders and an in-process scripted judge used by the test suite."""

import json

import httpx

from agent_kernel import JevSupervisorEngine, ProposedAction
from agent_kernel.models import JevChoiceOutput, JevNoulOutput, JevParallelResponse, JevScoreOutput


def choice(selected: str = "read_file", conf: float = 0.95) -> dict:
    return {
        "type": "choice",
        "content": {"selected_option": selected, "confidence": conf, "reasoning": "r"},
    }


def noul(prob: float = 0.10) -> dict:
    return {"type": "noul", "content": {"probability": prob, "reasoning": "r"}}


def score(value: float = 1.0) -> dict:
    return {"type": "score", "content": {"score": value, "reasoning": "r"}}


def payload(selected: str = "read_file", conf: float = 0.95, prob: float = 0.10,
            score_value: float = 1.0) -> dict:
    return {"choices": [choice(selected, conf), noul(prob), score(score_value)]}


def make_action(tool_name: str = "read_file") -> ProposedAction:
    return ProposedAction(
        tool_name=tool_name,
        arguments={"path": "/test"},
        rationale="testing",
        current_task="test",
        execution_history=[],
    )


def make_response(selected: str = "read_file", conf: float = 0.95, prob: float = 0.10,
                  blast: float = 1.0) -> JevParallelResponse:
    return JevParallelResponse(
        valid_tool=JevChoiceOutput(selected_option=selected, confidence=conf, reasoning=""),
        is_destructive=JevNoulOutput(probability=prob, reasoning=""),
        blast_radius=JevScoreOutput(score=blast, reasoning=""),
    )


class ScriptedJudge:
    """httpx.MockTransport handler replaying scripted judge responses.

    Each script item is one of:
      * ``dict`` / ``bytes``                     -> HTTP 200 with that body
      * ``(status, dict | bytes | exception)``   -> that status/body, or the
        exception raised instead of responding
      * an exception class / instance            -> raised (status ignored)

    When the script is exhausted, ``default`` is served with HTTP 200.
    Every incoming request is recorded in ``requests`` with its decoded
    JSON body in ``bodies``.
    """

    def __init__(self, script: list | None = None, default: dict | bytes | None = None) -> None:
        self.script: list = list(script or [])
        self.default = payload() if default is None else default
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict | None] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(request.content) if request.content else None)
        item = self.script.pop(0) if self.script else (200, self.default)
        if not isinstance(item, tuple):
            item = (200, item)
        status, body = item
        if isinstance(body, type) and issubclass(body, Exception):
            try:
                exc = body("injected by ScriptedJudge", request=request)
            except TypeError:
                exc = body()
            raise exc
        if isinstance(body, Exception):
            raise body
        if isinstance(body, bytes):
            return httpx.Response(status, content=body, request=request)
        return httpx.Response(status, json=body, request=request)


def make_engine(script: list | None = None, **kwargs) -> tuple[JevSupervisorEngine, ScriptedJudge]:
    """Build an engine wired to an in-process ScriptedJudge transport."""
    judge = ScriptedJudge(script)
    kwargs.setdefault("api_base_url", "http://judge.test")
    engine = JevSupervisorEngine(**kwargs)
    engine._client = httpx.AsyncClient(transport=httpx.MockTransport(judge))
    return engine, judge
