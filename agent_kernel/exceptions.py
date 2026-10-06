from agent_kernel.models import DecisionResult


class AgentKernelError(Exception):
    """Base class for every error raised by agent_kernel."""


class ResponseParseError(AgentKernelError):
    """The judge's response could not be parsed into a JevParallelResponse."""


class ActionBlocked(AgentKernelError):
    """Raised by execute() when the policy verdict is not ALLOW."""

    def __init__(self, decision: DecisionResult) -> None:
        super().__init__(f"action {decision.verdict}: {decision.reason}")
        self.decision = decision
