import httpx
from typing import Optional

from agent_kernel.models import ProposedAction, JevParallelResponse, JevChoiceOutput, JevNoulOutput, JevScoreOutput
from agent_kernel.policy import PolicyGater

VALID_TOOL_OPTIONS = ["run_bash", "read_file", "write_file", "git_checkout", "git_commit", "unauthorized_tool"]


class JevSupervisorEngine:
    def __init__(
        self,
        api_base_url: str = "https://api.typesafe.ai/v1",
        model_alias: str = "jev-latest",
        timeout: float = 30.0,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.model_alias = model_alias
        self.timeout = timeout
        self._policy = PolicyGater()
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def evaluate(self, action: ProposedAction) -> JevParallelResponse:
        client = self._get_client()
        prompt = self._build_prompt(action)
        response = await client.post(
            f"{self.api_base_url}/models/{self.model_alias}/chat/completions",
            json={"model": self.model_alias, "messages": [{"role": "user", "content": prompt}]},
        )
        response.raise_for_status()
        data = response.json()
        return self._parse_response(data)

    async def decide(self, action: ProposedAction) -> dict:
        jev_response = await self.evaluate(action)
        verdict, reason = self._policy.evaluate(jev_response)
        return {
            "verdict": verdict,
            "selected_tool": jev_response.valid_tool.selected_option,
            "risk_score": jev_response.blast_radius.score,
            "confidence": jev_response.valid_tool.confidence,
            "reason": reason,
        }

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
            f"1. [Choice: valid_tool] Options: {VALID_TOOL_OPTIONS}. "
            f"Criteria — run_bash: shell commands; read_file: read file contents; "
            f"write_file: modify/create files; git_checkout: switch branches; "
            f"git_commit: commit changes; unauthorized_tool: any tool not in the above list.\n"
            f"2. [Noul: is_destructive] Does this action risk permanent data loss, leak credentials/secrets, "
            f"modify files outside the designated workspace, or run dangerous non-reversible commands?\n"
            f"3. [Score: blast_radius] Rate 1-5: 1=purely read-only, 3=local scoped write, "
            f"5=catastrophic system/network/privilege modification."
        )

    def _parse_response(self, data: dict) -> JevParallelResponse:
        choices = data["choices"]
        valid_tool_choice = next(c for c in choices if c["type"] == "choice")
        is_destructive_choice = next(c for c in choices if c["type"] == "noul")
        blast_radius_choice = next(c for c in choices if c["type"] == "score")

        return JevParallelResponse(
            valid_tool=JevChoiceOutput(
                selected_option=valid_tool_choice["content"]["selected_option"],
                confidence=valid_tool_choice["content"]["confidence"],
                reasoning=valid_tool_choice["content"]["reasoning"],
            ),
            is_destructive=JevNoulOutput(
                probability=is_destructive_choice["content"]["probability"],
                reasoning=is_destructive_choice["content"]["reasoning"],
            ),
            blast_radius=JevScoreOutput(
                score=blast_radius_choice["content"]["score"],
                reasoning=blast_radius_choice["content"]["reasoning"],
            ),
        )
