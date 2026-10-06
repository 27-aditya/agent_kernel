import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from tests.helpers import ScriptedJudge


class _JudgeHandler(BaseHTTPRequestHandler):
    judge: ScriptedJudge

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        request = httpx.Request(
            "POST",
            f"http://judge.test{self.path}",
            content=raw,
            headers=[(k, v) for k, v in self.headers.items()],
        )
        response = self.judge(request)
        data = response.content
        self.send_response(response.status_code)
        self.send_header(
            "Content-Type", response.headers.get("content-type", "application/json")
        )
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # silence request logging
        pass


@pytest.fixture
def judge_server():
    """A real localhost judge server backed by a ScriptedJudge."""
    judge = ScriptedJudge()
    handler = type("BoundHandler", (_JudgeHandler,), {"judge": judge})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", judge
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
