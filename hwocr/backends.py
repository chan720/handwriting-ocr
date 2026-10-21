"""Model backends. All run fully offline once the model weights are on your machine.

- OllamaBackend:       native Ollama /api/chat (supports `think: false`, `num_ctx`).
- OpenAIBackend:       OpenAI-compatible /v1/chat/completions (vLLM, llama.cpp server, LM Studio, SGLang).
- TransformersBackend: Hugging Face transformers in-process.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass

import requests
from PIL import Image

from .config import BackendConfig, Config, TransformersConfig

log = logging.getLogger(__name__)


@dataclass
class Generation:
    text: str
    truncated: bool  # hit max_tokens
    seconds: float


class BackendError(RuntimeError):
    pass


def image_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=92)
    else:
        img.save(buf, format=fmt, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Backend:
    name = "base"

    def generate(self, image: Image.Image, prompt: str, system: str, repetition_penalty: float | None = None) -> Generation:
        raise NotImplementedError

    def check(self) -> str:
        """Return a human-readable status string, raise BackendError if unusable."""
        return "ok"

    def model_warning(self) -> str | None:
        """A warning about the configured model (e.g. an always-thinking variant), or None."""
        return None


class _HTTPBackend(Backend):
    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self.session = requests.Session()

    def _post(self, url: str, payload: dict, retries: int = 2) -> dict:
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                r = self.session.post(url, json=payload, timeout=self.cfg.timeout,
                                      headers={"Authorization": f"Bearer {self.cfg.api_key}"})
                if r.status_code >= 400:
                    raise BackendError(f"HTTP {r.status_code} from {url}: {r.text[:500]}")
                return r.json()
            except requests.ConnectionError as e:
                raise BackendError(
                    f"Cannot reach model server at {url}. Is it running? ({e.__class__.__name__})"
                ) from e
            except requests.ReadTimeout as e:
                # The server accepted the request but produced nothing in time. Re-sending would only
                # queue the same work behind the stuck request, so fail fast with a useful message.
                raise BackendError(
                    f"No response from the model server within {self.cfg.timeout:.0f}s. Either the server is stuck "
                    "(check its window/log for errors such as out-of-memory) or this machine needs longer: "
                    "raise it with --timeout, or use a smaller model / --max-side."
                ) from e
            except (requests.ConnectTimeout, requests.exceptions.ChunkedEncodingError) as e:
                last_err = e
                log.warning("Request failed (%s), retry %d/%d", e.__class__.__name__, attempt + 1, retries)
                time.sleep(2 ** attempt)
        raise BackendError(f"Request to {url} failed after retries: {last_err}")


class OllamaBackend(_HTTPBackend):
    name = "ollama"

    def __init__(self, cfg: BackendConfig):
        super().__init__(cfg)
        self.base = cfg.base_url.rstrip("/").removesuffix("/v1")
        self._think_supported = True

    def generate(self, image, prompt, system, repetition_penalty=None) -> Generation:
        payload = {
            "model": self.cfg.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt, "images": [image_to_b64(image)]},
            ],
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
                "num_ctx": self.cfg.num_ctx,
                "repeat_penalty": repetition_penalty or self.cfg.repetition_penalty,
            },
        }
        if self.cfg.num_thread > 0:
            payload["options"]["num_thread"] = self.cfg.num_thread
        if self.cfg.num_gpu >= 0:
            payload["options"]["num_gpu"] = self.cfg.num_gpu
        if self.cfg.disable_thinking and self._think_supported:
            payload["think"] = False
        t0 = time.time()
        try:
            data = self._post(f"{self.base}/api/chat", payload)
        except BackendError as e:
            # Older Ollama versions / non-thinking models may reject the `think` field.
            if "think" in payload and "think" in str(e).lower():
                self._think_supported = False
                payload.pop("think")
                data = self._post(f"{self.base}/api/chat", payload)
            else:
                raise
        msg = data.get("message") or {}
        return Generation(
            text=msg.get("content", ""),
            truncated=data.get("done_reason") == "length",
            seconds=time.time() - t0,
        )

    def check(self) -> str:
        try:
            r = self.session.get(f"{self.base}/api/tags", timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            raise BackendError(f"Ollama not reachable at {self.base}: {e}. Start it with `ollama serve`.") from e
        names = {m.get("name") for m in r.json().get("models", [])} | {m.get("model") for m in r.json().get("models", [])}
        want = self.cfg.model
        if want not in names and f"{want}:latest" not in names:
            raise BackendError(f"Model '{want}' is not pulled. Run: ollama pull {want}\nAvailable: {sorted(n for n in names if n)}")
        msg = f"Ollama at {self.base}, model '{want}' available"
        warning = self.model_warning()
        return f"{msg}\nWARNING: {warning}" if warning else msg

    def model_warning(self) -> str | None:
        try:
            r = self.session.post(f"{self.base}/api/show", json={"model": self.cfg.model}, timeout=10)
            caps = r.json().get("capabilities", []) if r.ok else []
        except (requests.RequestException, ValueError):
            return None
        if "thinking" in caps:
            base = self.cfg.model.split(":")[0]
            return (f"'{self.cfg.model}' is a THINKING model: it writes long hidden reasoning before answering, "
                    "which on a CPU can take longer than the timeout for a single page. For OCR use an instruct "
                    f"variant instead (see https://ollama.com/library/{base}/tags, e.g. '{base}:8b-instruct').")
        return None


class OpenAIBackend(_HTTPBackend):
    name = "openai"

    def __init__(self, cfg: BackendConfig):
        super().__init__(cfg)
        base = cfg.base_url.rstrip("/")
        self.base = base if base.endswith("/v1") else base + "/v1"

    def generate(self, image, prompt, system, repetition_penalty=None) -> Generation:
        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_b64(image)}"}},
                    {"type": "text", "text": prompt},
                ]},
            ],
        }
        rp = repetition_penalty or self.cfg.repetition_penalty
        if rp and rp != 1.0:
            payload["repetition_penalty"] = rp  # vLLM / llama.cpp / SGLang extension
        if self.cfg.disable_thinking:
            # Understood by vLLM, SGLang and llama.cpp for Qwen3-family templates; ignored elsewhere.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload.update(self.cfg.extra_body or {})
        t0 = time.time()
        data = self._post(f"{self.base}/chat/completions", payload)
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError) as e:
            raise BackendError(f"Unexpected response: {str(data)[:300]}") from e
        content = choice.get("message", {}).get("content") or ""
        return Generation(text=content, truncated=choice.get("finish_reason") == "length", seconds=time.time() - t0)

    def check(self) -> str:
        try:
            r = self.session.get(f"{self.base}/models", timeout=10,
                                 headers={"Authorization": f"Bearer {self.cfg.api_key}"})
            r.raise_for_status()
        except requests.RequestException as e:
            raise BackendError(f"OpenAI-compatible server not reachable at {self.base}: {e}") from e
        ids = [m.get("id") for m in r.json().get("data", [])]
        if ids and self.cfg.model not in ids:
            raise BackendError(f"Server does not serve model '{self.cfg.model}'. It serves: {ids}")
        return f"Server at {self.base} serving {ids or 'unknown models'}"


class TransformersBackend(Backend):
    name = "transformers"

    def __init__(self, cfg: BackendConfig, tcfg: TransformersConfig):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            raise BackendError("transformers backend needs: pip install -r requirements-transformers.txt") from e
        self.cfg, self.tcfg, self.torch = cfg, tcfg, torch
        kwargs: dict = {"device_map": tcfg.device_map, "dtype": tcfg.dtype}
        if tcfg.attn_implementation:
            kwargs["attn_implementation"] = tcfg.attn_implementation
        if tcfg.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        log.info("Loading %s (first run downloads the weights; afterwards it is fully offline)", tcfg.model_id)
        self.processor = AutoProcessor.from_pretrained(tcfg.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(tcfg.model_id, **kwargs).eval()

    def generate(self, image, prompt, system, repetition_penalty=None) -> Generation:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
        ]
        template_kwargs = {"enable_thinking": False} if self.cfg.disable_thinking else {}
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
            **template_kwargs,
        ).to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": self.cfg.max_tokens,
            "repetition_penalty": repetition_penalty or self.cfg.repetition_penalty,
        }
        if self.cfg.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.cfg.temperature)
        else:
            gen_kwargs["do_sample"] = False
        t0 = time.time()
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True)
        return Generation(text=text, truncated=len(new_tokens) >= self.cfg.max_tokens, seconds=time.time() - t0)

    def check(self) -> str:
        return f"Loaded {self.tcfg.model_id} on {self.model.device}"


def make_backend(cfg: Config) -> Backend:
    kind = cfg.backend.type.lower()
    if kind == "ollama":
        return OllamaBackend(cfg.backend)
    if kind in ("openai", "vllm", "llamacpp", "llama.cpp", "lmstudio", "sglang"):
        return OpenAIBackend(cfg.backend)
    if kind in ("transformers", "hf"):
        return TransformersBackend(cfg.backend, cfg.transformers)
    raise ValueError(f"Unknown backend type: {cfg.backend.type}")
