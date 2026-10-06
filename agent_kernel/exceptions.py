class AgentKernelError(Exception):
    """Base class for every error raised by agent_kernel."""


class ResponseParseError(AgentKernelError):
    """The judge's response could not be parsed into a JevParallelResponse."""
