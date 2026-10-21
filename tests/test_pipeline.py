import json

from conftest import PAGE_TEXT, render_page

from hwocr.backends import Generation, OllamaBackend, OpenAIBackend
from hwocr.cli import main
from hwocr.config import load_config
from hwocr.pipeline import Pipeline


class ScriptedBackend:
    """In-memory backend returning queued replies; records prompts."""

    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, image, prompt, system, repetition_penalty=None):
        self.calls.append({"prompt": prompt, "size": image.size, "rp": repetition_penalty})
        text, truncated = self.replies.pop(0) if self.replies else ("", False)
        return Generation(text, truncated, 0.01)


def _cfg(**over):
    cfg = load_config(None)
    for k, v in over.items():
        section, key = k.split("__")
        setattr(getattr(cfg, section), key, v)
    return cfg


def test_full_page_success(tmp_path, page_image):
    page_image.save(tmp_path / "letter.png")
    backend = ScriptedBackend([(PAGE_TEXT, False)])
    docs = Pipeline(_cfg(), backend).run([tmp_path / "letter.png"], tmp_path / "out")
    assert docs[0].pages[0].strategy == "full"
    assert (tmp_path / "out" / "letter.txt").read_text().strip() == PAGE_TEXT
    data = json.loads((tmp_path / "out" / "letter.json").read_text())
    assert [p["lines"] for p in data["pages"][0]["paragraphs"]][0] == [
        "Dear Sam,", "thank you for the letter you sent", "last week from the coast."]
    md = (tmp_path / "out" / "letter.md").read_text()
    assert "Dear Sam,  \nthank you" in md


def test_loop_retry_then_fallback_to_strips(page_image):
    loop = "Dear Sam,\n" + "thank you\n" * 30
    backend = ScriptedBackend([
        (loop, False),                      # full page loops
        (loop, False),                      # retry with higher penalty still loops
        ("Dear Sam,\nthank you for the letter you sent\nlast week from the coast.", False),  # strip 1
        ("The weather here has been cold\nand the garden is still asleep.\n\nWith love, Ana", False),  # strip 2
    ])
    text, strategy, flags = Pipeline(_cfg(), backend).transcribe_image(page_image)
    assert strategy == "strips:2"
    assert "fallback_to_strips" in flags and "repetition_retry" in flags
    assert backend.calls[1]["rp"] >= 1.15
    assert "strip of a taller page" in backend.calls[2]["prompt"]
    assert "With love, Ana" in text and text.count("thank you") == 1


def test_truncation_triggers_strips(page_image):
    backend = ScriptedBackend([("partial", True), ("top half", False), ("bottom half", False)])
    text, strategy, flags = Pipeline(_cfg(), backend).transcribe_image(page_image)
    assert strategy == "strips:2" and "truncated" in flags
    assert text == "top half\nbottom half"


def test_refine_accepted_and_rejected(page_image):
    backend = ScriptedBackend([("Dear Sarn,\nthank you", False), ("Dear Sam,\nthank you", False)])
    text, _, flags = Pipeline(_cfg(refine__enabled=True), backend).transcribe_image(page_image)
    assert text == "Dear Sam,\nthank you" and "refined" in flags
    assert "<draft>\nDear Sarn," in backend.calls[1]["prompt"]

    backend = ScriptedBackend([("a long faithful transcription of the page", False), ("summary", False)])
    text, _, flags = Pipeline(_cfg(refine__enabled=True), backend).transcribe_image(page_image)
    assert text == "a long faithful transcription of the page" and "refine_rejected" in flags


def test_cache_reused_and_invalidated(tmp_path, page_image):
    page_image.save(tmp_path / "a.png")
    out = tmp_path / "out"
    Pipeline(_cfg(), ScriptedBackend([("first", False)])).run([tmp_path / "a.png"], out)
    b = ScriptedBackend([("second", False)])
    Pipeline(_cfg(), b).run([tmp_path / "a.png"], out)
    assert b.calls == [] and (out / "a.txt").read_text().strip() == "first"
    # changing the model invalidates the cache
    b = ScriptedBackend([("third", False)])
    Pipeline(_cfg(backend__model="other"), b).run([tmp_path / "a.png"], out)
    assert (out / "a.txt").read_text().strip() == "third"


def test_ollama_backend_http(fake_server, page_image):
    srv = fake_server(lambda path, body, n: ("<think></think>" + PAGE_TEXT, "stop"))
    cfg = _cfg(backend__base_url=srv.url)
    backend = OllamaBackend(cfg.backend)
    assert "available" in backend.check()
    text, strategy, _ = Pipeline(cfg, backend).transcribe_image(page_image)
    assert text == PAGE_TEXT and strategy == "full"
    path, body = srv.requests[0]
    assert path == "/api/chat" and body["think"] is False
    assert body["options"]["num_ctx"] == 16384 and body["options"]["temperature"] == 0.0
    assert len(body["messages"][1]["images"]) == 1
    assert "num_thread" not in body["options"]

    assert "num_gpu" not in body["options"]

    cfg.backend.num_thread, cfg.backend.num_gpu = 4, 12
    OllamaBackend(cfg.backend).generate(page_image, "p", "s")
    assert srv.requests[-1][1]["options"]["num_thread"] == 4
    assert srv.requests[-1][1]["options"]["num_gpu"] == 12


def test_ollama_think_rejected_retries_without(fake_server, page_image):
    from http.server import BaseHTTPRequestHandler  # noqa: F401  (server built in conftest)

    def reply(path, body, n):
        return ("ok text", "stop")

    srv = fake_server(reply)
    cfg = _cfg(backend__base_url=srv.url)
    backend = OllamaBackend(cfg.backend)
    original = backend._post

    def flaky(url, payload, retries=2):
        if "think" in payload:
            from hwocr.backends import BackendError
            raise BackendError('HTTP 400: "qwen3-vl:8b" does not support thinking')
        return original(url, payload, retries)

    backend._post = flaky
    assert backend.generate(page_image, "p", "s").text == "ok text"
    assert backend._think_supported is False


def test_openai_backend_http_and_cli(fake_server, tmp_path, page_image):
    srv = fake_server(lambda path, body, n: (PAGE_TEXT, "stop"))
    cfg = _cfg(backend__base_url=srv.url, backend__type="openai", backend__model="Qwen/Qwen3-VL-8B-Instruct")
    assert "serving" in OpenAIBackend(cfg.backend).check()

    page_image.save(tmp_path / "doc.jpg")
    gt = tmp_path / "gt"
    gt.mkdir()
    (gt / "doc.txt").write_text(PAGE_TEXT)
    rc = main(["run", str(tmp_path / "doc.jpg"), "-o", str(tmp_path / "out"), "--backend", "openai",
               "--base-url", srv.url, "--model", "Qwen/Qwen3-VL-8B-Instruct", "--language", "English"])
    assert rc == 0
    path, body = srv.requests[-1]
    assert path == "/v1/chat/completions"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "written in English" in body["messages"][1]["content"][1]["text"]
    assert main(["eval", "--pred", str(tmp_path / "out"), "--gt", str(gt)]) == 0


def test_bench_ranks_models(fake_server, tmp_path, capsys):
    def reply(path, body, n):
        good = body["model"] == "good-model"
        return (PAGE_TEXT if good else PAGE_TEXT.replace("garden", "gordon").replace("\n\n", "\n"), "stop")

    srv = fake_server(reply)
    inputs, gt = tmp_path / "in", tmp_path / "gt"
    inputs.mkdir(), gt.mkdir()
    render_page().save(inputs / "p1.png")
    (gt / "p1.txt").write_text(PAGE_TEXT)
    rc = main(["bench", "--inputs", str(inputs), "--gt", str(gt), "--models", "bad-model,good-model",
               "-o", str(tmp_path / "bench"), "--backend", "openai", "--base-url", srv.url])
    assert rc == 0
    board = (tmp_path / "bench" / "leaderboard.md").read_text().splitlines()
    assert board[2].startswith("| good-model | 100.0%")
    assert board[3].startswith("| bad-model")


def test_read_timeout_fails_fast_without_retry(fake_server, page_image):
    import time as _t

    def slow(path, body, n):
        _t.sleep(2)
        return ("late", "stop")

    srv = fake_server(slow)
    cfg = _cfg(backend__base_url=srv.url, backend__timeout=0.5)
    import pytest
    from hwocr.backends import BackendError
    with pytest.raises(BackendError, match="No response"):
        OllamaBackend(cfg.backend).generate(page_image, "p", "s")
    _t.sleep(2.1)
    assert len(srv.requests) == 1  # no retry piled on top of the stuck request


def test_ollama_warns_about_thinking_models(fake_server):
    srv = fake_server(lambda path, body, n: ("x", "stop"))
    assert OllamaBackend(_cfg(backend__base_url=srv.url, backend__model="qwen3-vl:8b-instruct").backend).model_warning() is None
    w = OllamaBackend(_cfg(backend__base_url=srv.url, backend__model="qwen3-vl:8b-thinking").backend).model_warning()
    assert "THINKING" in w and "qwen3-vl:8b-instruct" in w
