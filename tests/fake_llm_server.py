"""A local stand-in for an OpenAI-compatible provider, so the real HTTP client is tested end to end."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Tuple

Reply = Tuple[int, Any, Dict[str, str]]


def chat_reply(content: str, prompt_tokens: int = 50, completion_tokens: int = 20) -> Reply:
    return 200, {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}, {}


class FakeProvider:
    def __init__(self, handler: Callable[[Dict[str, Any]], Reply]) -> None:
        self.handler = handler
        self.requests: List[Dict[str, Any]] = []
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                provider.requests.append({"path": self.path, "body": body,
                                          "authorization": self.headers.get("Authorization")})
                status, payload, headers = provider.handler(body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
