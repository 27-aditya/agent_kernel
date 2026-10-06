import asyncio
import inspect
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

import httpx
from pydantic import ValidationError

from agent_kernel.exceptions import ActionBlocked, ResponseParseError
from agent_kernel.models import (
    VALID_TOOLS,
    AuditEntry,
    DecisionResult,
    EscalationRequest,
    JevChoiceOutput,
    JevNoulOutput,
    JevParallelResponse,
    JevScoreOutput,
    ProposedAction,
    Verdict,
)
from agent_kernel.policy import PolicyGater

logger = logging.getLogger("agent_kernel.engine")

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

_VERDICT_LOG_LEVELS = {"ALLOW": logging.INFO, "ESCALATE": logging.WARNING, "BLOCK": logging.ERROR}

EscalationHandler = Callable[[EscalationRequest], "Verdict | Awaitable[Verdict]"]
DecisionHook = Callable[[AuditEntry], None]
Executor = Callable[[ProposedAction], "Any | Awaitable[Any]"]


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
        on_decision: Optional[DecisionHook] = None,
        escalation_handler: Optional[EscalationHandler] = None,
        audit_history_limit: int = 1000,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.model_alias = model_alias
        self.timeout = timeout
        self.api_key = api_key if api_key is not None else os.environ.get("AGENT_KERNEL_API_KEY")
        self.max_retries = max(0, max_retries)
        self.backoff_base = backoff_base
        self.extra_body = dict(extra_body or {})
        self._policy = policy or PolicyGater()
        self._on_decision = on_decision
        self._escalation_handler = escalation_handler
        self._audit: deque[AuditEntry] = deque(maxlen=max(0, audit_history_limit))
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def audit(self) -> tuple[AuditEntry, ...]:
        """Most recent decision records, oldest first (bounded by audit_history_limit)."""
        return tuple(self._audit)

    def clear_audit(self) -> None:
        self._audit.clear()

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

    async def evaluate(
        self,
        action: ProposedAction,
        request_id: Optional[str] = None,
    ) -> JevParallelResponse:
        request_id = request_id or uuid.uuid4().hex
        logger.debug(
            "judge request %s started for tool=%s", request_id, action.tool_name,
            extra={"request_id": request_id, "tool_name": action.tool_name,
                   "model_alias": self.model_alias},
        )
        prompt = self._build_prompt(action)
        payload: dict[str, Any] = {
            "model": self.model_alias,
            "messages": [{"role": "user", "content": prompt}],
            **self.extra_body,
        }
        response = await self._post_with_retry(payload, request_id)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            logger.error(
                "judge request %s failed with HTTP %s", request_id, response.status_code,
                extra={"request_id": request_id, "status_code": response.status_code,
                       "tool_name": action.tool_name},
            )
            raise
        try:
            data = response.json()
        except ValueError as exc:
            logger.error(
                "judge request %s returned a non-JSON body: %s", request_id, exc,
                extra={"request_id": request_id, "tool_name": action.tool_name},
            )
            raise ResponseParseError("judge response body is not valid JSON") from exc
        try:
            return self._parse_response(data)
        except ResponseParseError as exc:
            logger.error(
                "judge request %s returned a malformed response: %s", request_id, exc,
                extra={"request_id": request_id, "tool_name": action.tool_name},
            )
            raise

    async def _post_with_retry(self, payload: dict[str, Any], request_id: str) -> httpx.Response:
        client = self._get_client()
        url = f"{self.api_base_url}/models/{self.model_alias}/chat/completions"
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(url, json=payload, headers=self._headers())
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt == self.max_retries:
                    raise
                logger.warning(
                    "judge request %s retry %d/%d after transport error: %s",
                    request_id, attempt + 1, self.max_retries, exc,
                    extra={"request_id": request_id, "attempt": attempt + 1,
                           "max_retries": self.max_retries},
                )
            else:
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    return response
                if attempt == self.max_retries:
                    return response
                logger.warning(
                    "judge request %s retry %d/%d after status=%s",
                    request_id, attempt + 1, self.max_retries, response.status_code,
                    extra={"request_id": request_id, "attempt": attempt + 1,
                           "max_retries": self.max_retries,
                           "status_code": response.status_code},
                )
            await asyncio.sleep(self.backoff_base * (2**attempt))
        raise AssertionError("unreachable: retry loop must return or raise")

    async def decide(self, action: ProposedAction) -> DecisionResult:
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        try:
            jev_response = await self.evaluate(action, request_id=request_id)
        except ResponseParseError as exc:
            decision = DecisionResult(
                verdict="ESCALATE",
                selected_tool=action.tool_name,
                risk_score=0.0,
                confidence=0.0,
                reason=f"ESCALATED: malformed judge response ({exc}) — requires human review.",
            )
        else:
            verdict, reason = self._policy.evaluate(jev_response, action)
            decision = DecisionResult(
                verdict=verdict,
                selected_tool=jev_response.valid_tool.selected_option,
                risk_score=jev_response.blast_radius.score,
                confidence=jev_response.valid_tool.confidence,
                reason=reason,
            )
        escalated = False
        if decision.verdict == "ESCALATE" and self._escalation_handler is not None:
            decision, escalated = await self._resolve_escalation(request_id, action, decision)
        self._record_audit(request_id, action, decision, time.perf_counter() - started, escalated)
        return decision

    async def _resolve_escalation(
        self,
        request_id: str,
        action: ProposedAction,
        decision: DecisionResult,
    ) -> tuple[DecisionResult, bool]:
        """Hand an ESCALATE verdict to the escalation handler and apply its resolution.

        The handler may be sync or async and must return ALLOW, ESCALATE, or BLOCK.
        Anything else — including a handler that raises — keeps the ESCALATE verdict.
        """
        request = EscalationRequest(
            request_id=request_id,
            raised_at=datetime.now(timezone.utc),
            action=action,
            decision=decision,
        )
        logger.warning(
            "decision %s escalated — awaiting escalation handler", request_id,
            extra={"request_id": request_id, "tool_name": action.tool_name,
                   "verdict": decision.verdict},
        )
        try:
            resolution = self._escalation_handler(request)
            if inspect.isawaitable(resolution):
                resolution = await resolution
        except Exception:
            logger.exception(
                "escalation handler raised for request %s; keeping ESCALATE", request_id
            )
            return decision, True
        if resolution in ("ALLOW", "BLOCK"):
            return (
                decision.model_copy(
                    update={
                        "verdict": resolution,
                        "reason": f"{decision.reason} Escalation resolved to {resolution}.",
                    }
                ),
                True,
            )
        if resolution != "ESCALATE":
            logger.error(
                "escalation handler returned %r for request %s; keeping ESCALATE",
                resolution, request_id,
                extra={"request_id": request_id},
            )
        return decision, True

    async def execute(self, action: ProposedAction, executor: Executor) -> tuple[DecisionResult, Any]:
        """Decide, then run ``executor(action)`` only if the verdict is ALLOW.

        Raises ``ActionBlocked`` (carrying the decision) for BLOCK and ESCALATE
        verdicts, so a guarded agent loop can never run unapproved work.
        ``executor`` may be sync or async.
        """
        decision = await self.decide(action)
        if decision.verdict != "ALLOW":
            raise ActionBlocked(decision)
        result = executor(action)
        if inspect.isawaitable(result):
            result = await result
        return decision, result

    def _record_audit(
        self,
        request_id: str,
        action: ProposedAction,
        decision: DecisionResult,
        latency_s: float,
        escalated: bool = False,
    ) -> None:
        entry = AuditEntry(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            model_alias=self.model_alias,
            latency_ms=latency_s * 1000.0,
            escalated=escalated,
            action=action,
            decision=decision,
        )
        self._audit.append(entry)

        logger.log(
            _VERDICT_LOG_LEVELS[decision.verdict],
            "decision %s: %s for tool=%s (risk=%.2f confidence=%.2f in %.1fms)",
            request_id, decision.verdict, action.tool_name,
            decision.risk_score, decision.confidence, entry.latency_ms,
            extra={
                "request_id": request_id,
                "verdict": decision.verdict,
                "selected_tool": decision.selected_tool,
                "tool_name": action.tool_name,
                "risk_score": decision.risk_score,
                "confidence": decision.confidence,
                "latency_ms": entry.latency_ms,
                "model_alias": self.model_alias,
                "escalated": escalated,
                "reason": decision.reason,
            },
        )

        if self._on_decision is not None:
            try:
                self._on_decision(entry)
            except Exception:
                logger.exception("on_decision hook raised for request %s", request_id)

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
