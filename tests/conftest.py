import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image, ImageDraw, ImageFont

PAGE_TEXT = (
    "Dear Sam,\n"
    "thank you for the letter you sent\n"
    "last week from the coast.\n"
    "\n"
    "The weather here has been cold\n"
    "and the garden is still asleep.\n"
    "\n"
    "With love, Ana"
)


def _font(size=34):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_page(text=PAGE_TEXT, size=(1240, 1754)) -> Image.Image:
    """A synthetic 'scanned page' with line and paragraph spacing."""
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    y = 150
    for line in text.split("\n"):
        if line:
            d.text((120, y), line, fill=(20, 20, 60), font=_font())
            y += 60
        else:
            y += 70
    return img


@pytest.fixture
def page_image():
    return render_page()


class FakeServer:
    """Minimal HTTP server mimicking Ollama /api/chat and OpenAI /v1/chat/completions."""

    def __init__(self, reply_fn):
        self.requests = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": "qwen3-vl:8b", "model": "qwen3-vl:8b"}]})
                elif self.path == "/v1/models":
                    self._send(200, {"data": [{"id": "Qwen/Qwen3-VL-8B-Instruct"}]})
                else:
                    self._send(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/show":
                    caps = ["completion", "vision"] + (["thinking"] if "thinking" in body["model"] else [])
                    self._send(200, {"capabilities": caps})
                    return
                server.requests.append((self.path, body))
                text, reason = reply_fn(self.path, body, len(server.requests))
                if self.path == "/api/chat":
                    self._send(200, {"message": {"role": "assistant", "content": text}, "done_reason": reason})
                elif self.path == "/v1/chat/completions":
                    self._send(200, {"choices": [{"message": {"content": text}, "finish_reason": reason}]})
                else:
                    self._send(404, {})

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def fake_server():
    servers = []

    def make(reply_fn):
        s = FakeServer(reply_fn)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()
