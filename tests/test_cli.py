import json
import subprocess
import sys

import pytest

from agent_kernel.cli import build_parser, main
from tests.helpers import payload


def run_cli(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured


def base_args(url: str) -> list[str]:
    return ["read_file", "--api-base-url", url, "--retries", "0"]


class TestCli:
    def test_allow_exits_zero_with_json_decision(self, capsys, judge_server):
        url, _ = judge_server
        code, out = run_cli(capsys, *base_args(url), "--args", '{"path": "/etc/hosts"}')
        assert code == 0
        decision = json.loads(out.out)
        assert decision["verdict"] == "ALLOW"
        assert decision["selected_tool"] == "read_file"
        assert decision["risk_score"] == 1.0
        assert decision["confidence"] == 0.95
        assert decision["reason"]

    def test_block_exits_four(self, capsys, judge_server):
        url, judge = judge_server
        judge.script.append((200, payload("read_file", 0.99, 0.95, 5.0)))
        code, out = run_cli(capsys, *base_args(url), "--args", '{"path": "/x"}')
        assert code == 4
        assert json.loads(out.out)["verdict"] == "BLOCK"

    def test_escalate_exits_three(self, capsys, judge_server):
        url, judge = judge_server
        judge.script.append((200, payload("read_file", 0.95, 0.60, 1.0)))
        code, out = run_cli(capsys, *base_args(url))
        assert code == 3
        assert json.loads(out.out)["verdict"] == "ESCALATE"

    def test_interactive_escalation_allow(self, capsys, judge_server, monkeypatch):
        url, judge = judge_server
        judge.script.append((200, payload("read_file", 0.95, 0.60, 1.0)))
        monkeypatch.setattr("builtins.input", lambda prompt="": "y")
        code, out = run_cli(capsys, *base_args(url), "--interactive")
        assert code == 0
        assert json.loads(out.out)["verdict"] == "ALLOW"
        assert "Escalation resolved to ALLOW" in json.loads(out.out)["reason"]

    def test_interactive_escalation_hold_keeps_escalate(self, capsys, judge_server, monkeypatch):
        url, judge = judge_server
        judge.script.append((200, payload("read_file", 0.95, 0.60, 1.0)))
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        code, out = run_cli(capsys, *base_args(url), "--interactive")
        assert code == 3
        assert json.loads(out.out)["verdict"] == "ESCALATE"

    def test_interactive_escalation_block(self, capsys, judge_server, monkeypatch):
        url, judge = judge_server
        judge.script.append((200, payload("read_file", 0.95, 0.60, 1.0)))
        monkeypatch.setattr("builtins.input", lambda prompt="": "b")
        code, out = run_cli(capsys, *base_args(url), "--interactive")
        assert code == 4
        assert json.loads(out.out)["verdict"] == "BLOCK"

    def test_interactive_eof_keeps_escalate(self, capsys, judge_server, monkeypatch):
        url, judge = judge_server
        judge.script.append((200, payload("read_file", 0.95, 0.60, 1.0)))

        def no_stdin(prompt=""):
            raise EOFError

        monkeypatch.setattr("builtins.input", no_stdin)
        code, _ = run_cli(capsys, *base_args(url), "--interactive")
        assert code == 3

    def test_invalid_json_args_exits_one(self, capsys, judge_server):
        url, _ = judge_server
        code, out = run_cli(capsys, *base_args(url), "--args", "{not json")
        assert code == 1
        assert "error:" in out.err

    def test_non_object_args_exits_one(self, capsys, judge_server):
        url, _ = judge_server
        code, out = run_cli(capsys, *base_args(url), "--args", "[1, 2]")
        assert code == 1
        assert "JSON object" in out.err

    def test_server_error_exits_one(self, capsys, judge_server):
        url, judge = judge_server
        judge.script.append((500, {"error": "down"}))
        code, out = run_cli(capsys, *base_args(url))
        assert code == 1
        assert "error:" in out.err

    def test_history_and_rationale_reach_the_judge(self, capsys, judge_server):
        url, judge = judge_server
        code, _ = run_cli(
            capsys, *base_args(url),
            "--rationale", "need the file",
            "--task", "fix the bug",
            "--history", "read_file /README.md",
            "--history", "run_bash ls",
        )
        assert code == 0
        prompt = judge.bodies[0]["messages"][0]["content"]
        assert "need the file" in prompt
        assert "fix the bug" in prompt
        assert "read_file /README.md" in prompt
        assert "run_bash ls" in prompt

    def test_help_exits_zero(self):
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args(["--help"])
        assert exc.value.code == 0

    def test_keyboard_interrupt_exits_130(self, capsys, monkeypatch, judge_server):
        url, _ = judge_server

        def interrupted(coroutine):
            coroutine.close()
            raise KeyboardInterrupt

        monkeypatch.setattr("agent_kernel.cli.asyncio.run", interrupted)
        code, _ = run_cli(capsys, *base_args(url))
        assert code == 130

    def test_module_invocation(self):
        result = subprocess.run(
            [sys.executable, "-m", "agent_kernel", "--help"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "agent-kernel" in result.stdout

    def test_module_interactive_keeps_stdout_json(self, judge_server):
        url, judge = judge_server
        judge.script.append((200, payload("read_file", 0.95, 0.60, 1.0)))
        result = subprocess.run(
            [sys.executable, "-m", "agent_kernel", "read_file",
             "--api-base-url", url, "--retries", "0", "--interactive"],
            input="y\n", capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        decision = json.loads(result.stdout)
        assert decision["verdict"] == "ALLOW"
        assert "ESCALATED" in result.stderr
        assert "allow, block, or hold" in result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
