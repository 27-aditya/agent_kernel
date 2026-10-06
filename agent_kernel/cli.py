"""Command-line interface: decide whether an agent action may run.

Exit codes:
    0  ALLOW
    1  error (bad input, network failure, judge failure)
    2  usage error (argparse)
    3  ESCALATE
    4  BLOCK
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from agent_kernel.engine import JevSupervisorEngine
from agent_kernel.models import DecisionResult, EscalationRequest, ProposedAction, Verdict

EXIT_ALLOW = 0
EXIT_ERROR = 1
EXIT_ESCALATE = 3
EXIT_BLOCK = 4

EXIT_CODES = {"ALLOW": EXIT_ALLOW, "ESCALATE": EXIT_ESCALATE, "BLOCK": EXIT_BLOCK}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-kernel",
        description="Ask the supervisor kernel whether an agent action may run.",
    )
    parser.add_argument("tool", help="proposed tool name (e.g. run_bash)")
    parser.add_argument(
        "--args", dest="arguments", default="{}", metavar="JSON",
        help="tool arguments as a JSON object (default: {})",
    )
    parser.add_argument("--rationale", default="", help="why the agent wants to run this")
    parser.add_argument("--task", dest="current_task", default="", help="the agent's current task")
    parser.add_argument(
        "--history", action="append", default=None, metavar="ENTRY",
        help="execution history entry (repeatable)",
    )
    parser.add_argument("--api-base-url", default="https://api.typesafe.ai/v1")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--api-key", default=None, help="defaults to AGENT_KERNEL_API_KEY")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--interactive", action="store_true",
        help="prompt for a human resolution when the verdict is ESCALATE",
    )
    return parser


def _interactive_resolution(request: EscalationRequest) -> Verdict:
    print(f"ESCALATED: {request.decision.reason}", file=sys.stderr)
    print(
        "Resolve escalation — allow, block, or hold? [y/N/a] ",
        end="", file=sys.stderr, flush=True,
    )
    try:
        answer = input().strip().lower()
    except EOFError:
        return "ESCALATE"
    if answer in ("y", "yes", "allow"):
        return "ALLOW"
    if answer in ("b", "block"):
        return "BLOCK"
    return "ESCALATE"


async def _decide(args: argparse.Namespace, action: ProposedAction) -> DecisionResult:
    engine = JevSupervisorEngine(
        api_base_url=args.api_base_url,
        model_alias=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=args.retries,
        escalation_handler=_interactive_resolution if args.interactive else None,
    )
    async with engine:
        return await engine.decide(action)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        arguments = json.loads(args.arguments)
        if not isinstance(arguments, dict):
            raise ValueError("--args must be a JSON object, e.g. '{\"path\": \"/etc/passwd\"}'")
        action = ProposedAction(
            tool_name=args.tool,
            arguments=arguments,
            rationale=args.rationale,
            current_task=args.current_task,
            execution_history=args.history or [],
        )
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    try:
        decision = asyncio.run(_decide(args, action))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(decision.model_dump_json(indent=2))
    return EXIT_CODES[decision.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
