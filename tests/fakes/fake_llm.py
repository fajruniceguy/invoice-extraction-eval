"""Fake OpenAI-compatible chat-completions server.

Used by the integration tests (real sockets, so httpx really talks HTTP to it) and for manual
smoke runs without an API key:

    python tests/fakes/fake_llm.py 8765 [--reject-schema]
    LLM_BASE_URL=http://127.0.0.1:8765/v1 LLM_MODEL=fake python scripts/smoke.py

Replies come from a scripted queue (`enqueue`); when the queue is empty the server answers
with a canned, valid extraction of tests/fixtures/sample_receipt.png. With
`reject_json_schema` the default reply to a `json_schema` request is a 400, like a server
that does not support structured outputs. Every request is recorded. This is a fake, not a
model: it only proves wiring.
"""

import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Self


@dataclass
class Reply:
    status: int = 200
    body: dict[str, Any] | str | None = None  # dict -> JSON, str -> raw text
    headers: dict[str, str] = field(default_factory=dict)
    delay_s: float = 0.0
    drop: bool = False  # close the connection without answering (a transport error)


@dataclass
class RecordedRequest:
    path: str
    headers: dict[str, str]
    body: dict[str, Any]
    received_at: float = field(default_factory=time.monotonic)  # when it arrived

    @property
    def response_format(self) -> str | None:
        return (self.body.get("response_format") or {}).get("type")


def _field(value: Any, evidence: str | None = None, confidence: float = 0.95) -> dict[str, Any]:
    return {
        "value": value,
        "confidence": confidence,
        "status": "extracted",
        "evidence": evidence or str(value),
        "page": 1,
        "reasons": [],
    }


def _absent(confidence: float = 0.95) -> dict[str, Any]:
    return {
        "value": None,
        "confidence": confidence,
        "status": "absent",
        "evidence": None,
        "page": None,
        "reasons": ["not stated on the document"],
    }


def _row(description: str, quantity: float, unit_price: float, amount: float) -> dict[str, Any]:
    return {
        "description": description,
        "quantity": quantity,
        "unit_price": unit_price,
        "amount": amount,
        "status": "extracted",
        "confidence": 0.95,
        "evidence": f"{description} {quantity} {unit_price} {amount}",
        "page": 1,
        "reasons": [],
    }


def canned_answer() -> dict[str, Any]:
    """A valid DocumentExtraction for the sample receipt (new three-way statuses)."""
    return {
        "vendor_name": _field("CORNER CAFE DEMO"),
        "vendor_address": _field("12 Example Street, Springfield"),
        "vendor_tax_id": _field("12-3456789", "Tax ID: 12-3456789"),
        "buyer_name": _absent(),
        "document_number": _field("R-0042", "Receipt No: R-0042"),
        "document_date": _field("2024-03-05", "Mar 5, 2024"),
        "due_date": _absent(),
        "currency": _field("USD", "$17.28", 0.7),
        "payment_method": _field("VISA ending 1234", "Paid by: VISA ending 1234"),
        "line_items": [
            _row("Latte", 2, 4.5, 9.0),
            _row("Croissant", 1, 3.25, 3.25),
            _row("Orange juice", 1, 3.75, 3.75),
        ],
        "subtotal": _field(16.0, "Subtotal 16.00"),
        "tax": _field(1.28, "Tax (8%) 1.28"),
        "discount": _absent(),
        "total": _field(17.28, "TOTAL $17.28"),
    }


def chat_reply(
    content: str | dict[str, Any] | None,
    *,
    prompt_tokens: int = 1500,
    completion_tokens: int = 700,
    usage: dict[str, Any] | None = None,
    finish_reason: str = "stop",
) -> Reply:
    """A 200 chat completion. `content` may be text, or a dict that is sent as JSON text."""
    text = json.dumps(content) if isinstance(content, dict) else content
    body: dict[str, Any] = {
        "choices": [
            {"message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}
        ],
        "usage": usage
        if usage is not None
        else {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return Reply(body=body)


class FakeLLMServer:
    def __init__(self, *, reject_json_schema: bool = False) -> None:
        self.reject_json_schema = reject_json_schema
        self.requests: list[RecordedRequest] = []
        self._script: deque[Reply] = deque()
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- scripting and inspection

    def enqueue(self, *replies: Reply) -> None:
        with self._lock:
            self._script.extend(replies)

    @property
    def base_url(self) -> str:
        assert self._server is not None, "server not started"
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    # -- lifecycle

    def start(self, port: int = 0) -> Self:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # keep test output clean
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                request = RecordedRequest(self.path, dict(self.headers.items()), body)
                reply = outer._next_reply(request)
                if reply.drop:
                    self.close_connection = True  # no response at all
                    return
                self._send(reply)

            def _send(self, reply: Reply) -> None:
                if reply.delay_s:
                    threading.Event().wait(reply.delay_s)
                payload = reply.body
                data = (payload if isinstance(payload, str) else json.dumps(payload or {})).encode()
                self.send_response(reply.status)
                content_type = "text/plain" if isinstance(payload, str) else "application/json"
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                for name, value in reply.headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        server.daemon_threads = True  # keep-alive connections must not block shutdown
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- behavior

    def _next_reply(self, request: RecordedRequest) -> Reply:
        with self._lock:
            self.requests.append(request)
            if self._script:
                return self._script.popleft()
        if self.reject_json_schema and request.response_format == "json_schema":
            return Reply(
                400, {"error": {"message": "response_format json_schema is not supported"}}
            )
        return chat_reply(canned_answer())


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = FakeLLMServer(reject_json_schema="--reject-schema" in sys.argv).start(port)
    print(f"fake LLM listening on {server.base_url}  (Ctrl+C to stop)", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
