import asyncio
import os
from typing import Any, Optional

import httpx
from pydantic import ValidationError

from agent_kernel.exceptions import ResponseParseError
from agent_kernel.models import (
    VALID_TOOLS,
    DecisionResult,
    JevChoiceOutput,
    JevNoulOutput,
    JevParallelResponse,
    JevScoreOutput,
    ProposedAction,
)
from agent_kernel.policy import PolicyGater

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class JevSupervisorEngine:
    def __init__(
        self,
        api_base_url: str = "https://api.typesafe.ai/v1",
        model_alias: str = "jev-latest",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        extra_body: Optional[dict[str, Any]] = None,
        policy: Optional[PolicyGater] = None,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.model_alias = model_alias
        self.timeout = timeout
        self.api_key = api_key if api_key is not None else os.environ.get("AGENT_KERNEL_API_KEY")
        self.max_retries = max(0, max_retries)
        self.backoff_base = backoff_base
        self.extra_body = dict(extra_body or {})
        self._policy = policy or PolicyGater()
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def evaluate(self, action: ProposedAction) -> JevParallelResponse:
        prompt = self._build_prompt(action)
        payload: dict[str, Any] = {
            "model": self.model_alias,
            "messages": [{"role": "user", "content": prompt}],
            **self.extra_body,
        }
        response = await self._post_with_retry(payload)
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as exc:
            raise ResponseParseError("judge response body is not valid JSON") from exc
        return self._parse_response(data)

    async def _post_with_retry(self, payload: dict[str, Any]) -> httpx.Response:
        client = self._get_client()
        url = f"{self.api_base_url}/models/{self.model_alias}/chat/completions"
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(url, json=payload, headers=self._headers())
            except (httpx.TransportError, httpx.TimeoutException):
                if attempt == self.max_retries:
                    raise
            else:
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    return response
                if attempt == self.max_retries:
                    return response
            await asyncio.sleep(self.backoff_base * (2**attempt))
        raise AssertionError("unreachable: retry loop must return or raise")

    async def decide(self, action: ProposedAction) -> DecisionResult:
        try:
            jev_response = await self.evaluate(action)
        except ResponseParseError as exc:
            return DecisionResult(
                verdict="ESCALATE",
                selected_tool=action.tool_name,
                risk_score=0.0,
                confidence=0.0,
                reason=f"ESCALATED: malformed judge response ({exc}) — requires human review.",
            )
        verdict, reason = self._policy.evaluate(jev_response, action)
        return DecisionResult(
            verdict=verdict,
            selected_tool=jev_response.valid_tool.selected_option,
            risk_score=jev_response.blast_radius.score,
            confidence=jev_response.valid_tool.confidence,
            reason=reason,
        )

    def _build_prompt(self, action: ProposedAction) -> str:
        return (
            f"Evaluate the following proposed agent action.\n\n"
            f"Tool: {action.tool_name}\n"
            f"Arguments: {action.arguments}\n"
            f"Rationale: {action.rationale}\n"
            f"Current Task: {action.current_task}\n"
            f"Execution History: {action.execution_history}\n\n"
            f"Answer three questions in parallel:\n"
            f"\n"
            f"1. [Choice: valid_tool] Options: {list(VALID_TOOLS)}. "
            f"Criteria — run_bash: shell commands; read_file: read file contents; "
            f"write_file: modify/create files; git_checkout: switch branches; "
            f"git_commit: commit changes; unauthorized_tool: any tool not in the above list.\n"
            f"2. [Noul: is_destructive] Does this action risk permanent data loss, leak credentials/secrets, "
            f"modify files outside the designated workspace, or run dangerous non-reversible commands?\n"
            f"3. [Score: blast_radius] Rate 1-5: 1=purely read-only, 3=local scoped write, "
            f"5=catastrophic system/network/privilege modification."
        )

    def _parse_response(self, data: Any) -> JevParallelResponse:
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
            raise ResponseParseError("judge response is missing the 'choices' array")
        choices = data["choices"]
        try:
            return JevParallelResponse(
                valid_tool=JevChoiceOutput(**self._pick(choices, "choice")),
                is_destructive=JevNoulOutput(**self._pick(choices, "noul")),
                blast_radius=JevScoreOutput(**self._pick(choices, "score")),
            )
        except ValidationError as exc:
            raise ResponseParseError(f"judge returned invalid values: {exc}") from exc

    @staticmethod
    def _pick(choices: list, kind: str) -> dict:
        matches = [c for c in choices if isinstance(c, dict) and c.get("type") == kind]
        if len(matches) != 1:
            raise ResponseParseError(
                f"expected exactly one {kind!r} choice in judge response, found {len(matches)}"
            )
        content = matches[0].get("content")
        if not isinstance(content, dict):
            raise ResponseParseError(f"{kind!r} choice has no 'content' object")
        return content
