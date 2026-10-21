"""End-to-end orchestration: ingest -> preprocess -> transcribe (full page / strips) -> refine -> merge -> save."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import reduce
from pathlib import Path

from PIL import Image

from .backends import Backend, make_backend
from .config import Config
from .ingest import Page, discover_inputs, iter_pages
from .postprocess import (clean_output, collapse_repetition, is_degenerate, length_ratio_ok, merge_strips,
                          to_paragraphs)
from .preprocess import preprocess, split_into_strips
from .prompts import SYSTEM_PROMPT, refine_prompt, transcribe_prompt

log = logging.getLogger(__name__)


@dataclass
class PageResult:
    page: int
    text: str
    strategy: str  # "full" or "strips:N"
    seconds: float
    flags: list[str] = field(default_factory=list)
    preprocess: dict = field(default_factory=dict)

    @property
    def paragraphs(self) -> list[list[str]]:
        return to_paragraphs(self.text)


@dataclass
class DocumentResult:
    source: str
    name: str
    pages: list[PageResult]
    model: str
    backend: str

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)


class Pipeline:
    def __init__(self, cfg: Config, backend: Backend | None = None):
        self.cfg = cfg
        self.backend = backend or make_backend(cfg)

    # ------------------------------------------------------------------ model calls
    def _ask(self, image: Image.Image, prompt: str, flags: list[str], retry: bool = True) -> tuple[str, bool, bool]:
        """One model call + cleaning. Retries once with a stronger repetition penalty if it loops."""
        gen = self.backend.generate(image, prompt, SYSTEM_PROMPT)
        text = clean_output(gen.text)
        degenerate = is_degenerate(text)
        if degenerate and retry:
            flags.append("repetition_retry")
            stronger = max(1.15, self.cfg.backend.repetition_penalty + 0.1)
            gen2 = self.backend.generate(image, prompt, SYSTEM_PROMPT, repetition_penalty=stronger)
            text2 = clean_output(gen2.text)
            if not is_degenerate(text2):
                return text2, gen2.truncated, False
        return text, gen.truncated, degenerate

    def _transcribe_unit(self, image: Image.Image, is_strip: bool, flags: list[str]) -> tuple[str, bool]:
        """Transcribe one image (page or strip), optionally refine. Returns (text, looks_bad)."""
        text, truncated, degenerate = self._ask(image, transcribe_prompt(self.cfg.prompt, is_strip), flags)
        bad = truncated or degenerate or not text
        if truncated:
            flags.append("truncated")
        if self.cfg.refine.enabled and text and not bad:
            revised, r_trunc, r_deg = self._ask(
                image, refine_prompt(self.cfg.prompt, text, is_strip), flags, retry=False
            )
            if revised and not r_trunc and not r_deg and length_ratio_ok(text, revised):
                if revised != text:
                    flags.append("refined")
                text = revised
            else:
                flags.append("refine_rejected")
        return text, bad

    def transcribe_image(self, image: Image.Image, debug_prefix: Path | None = None) -> tuple[str, str, list[str]]:
        """Transcribe one preprocessed page image. Returns (text, strategy, flags)."""
        flags: list[str] = []
        mode = self.cfg.tiling.mode
        if mode != "always":
            text, bad = self._transcribe_unit(image, False, flags)
            if not bad:
                return text, "full", flags
            if mode == "off":
                flags.append("degenerate_output")
                return collapse_repetition(text), "full", flags
            flags.append("fallback_to_strips")

        strips = split_into_strips(image, max(2, self.cfg.tiling.strips), self.cfg.tiling.overlap)
        texts = []
        for i, strip in enumerate(strips, 1):
            if debug_prefix:
                strip.save(f"{debug_prefix}_strip{i}.png")
            t, bad = self._transcribe_unit(strip, True, flags)
            texts.append(collapse_repetition(t) if bad else t)
        return reduce(merge_strips, texts) if texts else "", f"strips:{len(strips)}", flags

    # ------------------------------------------------------------------ pages / documents
    def fingerprint(self) -> str:
        """Hash of every setting that changes the output, so cached pages are reused only when valid."""
        c = self.cfg
        relevant = {
            "backend": [c.backend.type, c.backend.model, c.backend.temperature, c.backend.max_tokens],
            "hf": c.transformers.model_id if c.backend.type in ("transformers", "hf") else None,
            "ingest": asdict(c.ingest), "pre": asdict(c.preprocess), "tiling": asdict(c.tiling),
            "refine": asdict(c.refine), "prompt": asdict(c.prompt),
            "prompt_text": transcribe_prompt(c.prompt) + SYSTEM_PROMPT,
        }
        return hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def process_page(self, page: Page, debug_dir: Path | None = None) -> PageResult:
        t0 = time.time()
        image, pre_info = preprocess(page.image, self.cfg.preprocess)
        prefix = None
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            prefix = debug_dir / f"page_{page.number:04d}"
            image.save(f"{prefix}.png")
        text, strategy, flags = self.transcribe_image(image, prefix)
        return PageResult(page.number, text, strategy, round(time.time() - t0, 2), flags, pre_info)

    def process_document(self, path: Path, name: str, out_dir: Path | None = None,
                         pages: str | None = None, overwrite: bool = False) -> DocumentResult:
        cache_dir = out_dir / ".cache" / name if out_dir else None
        debug_dir = out_dir / "debug" / name if (out_dir and self.cfg.output.save_debug_images) else None
        fp = self.fingerprint()
        results: dict[int, PageResult] = {}
        workers = max(1, int(self.cfg.runtime.workers))

        def cached(n: int) -> PageResult | None:
            if not cache_dir or overwrite:
                return None
            f = cache_dir / f"page_{n:04d}.json"
            if f.exists():
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.pop("fingerprint", None) == fp:
                    return PageResult(**data)
            return None

        def work(page: Page) -> PageResult:
            res = self.process_page(page, debug_dir)
            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
                (cache_dir / f"page_{page.number:04d}.json").write_text(
                    json.dumps({**asdict(res), "fingerprint": fp}, ensure_ascii=False, indent=1), encoding="utf-8")
            note = f" [{res.strategy}{', ' + ', '.join(res.flags) if res.flags else ''}]"
            log.info("%s page %d: %d chars in %.1fs%s", name, page.number, len(res.text), res.seconds, note)
            return res

        with ThreadPoolExecutor(max_workers=workers) as pool:
            inflight: list[Future] = []
            for page in iter_pages(path, dpi=self.cfg.ingest.pdf_dpi, pages=pages):
                hit = cached(page.number)
                if hit:
                    log.info("%s page %d: cached", name, page.number)
                    results[page.number] = hit
                    continue
                inflight.append(pool.submit(work, page))
                if len(inflight) >= workers * 2:  # bound memory on long PDFs
                    r = inflight.pop(0).result()
                    results[r.page] = r
            for fut in inflight:
                r = fut.result()
                results[r.page] = r

        model = self.cfg.transformers.model_id if self.cfg.backend.type in ("transformers", "hf") else self.cfg.backend.model
        return DocumentResult(str(path), name, [results[k] for k in sorted(results)], model, self.cfg.backend.type)

    def run(self, inputs: list[str | Path], out_dir: str | Path, pages: str | None = None,
            overwrite: bool = False) -> list[DocumentResult]:
        from .output import write_outputs

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = discover_inputs(inputs)
        warning = self.backend.model_warning() if hasattr(self.backend, "model_warning") else None
        if warning:
            log.warning(warning)
        if not files:
            log.warning("No supported files found in %s", inputs)
        names = unique_names(files)
        docs = []
        for path in files:
            log.info("Processing %s", path)
            doc = self.process_document(path, names[path], out_dir, pages, overwrite)
            for written in write_outputs(doc, out_dir, self.cfg):
                log.info("  wrote %s", written)
            docs.append(doc)
        return docs


def unique_names(files: list[Path]) -> dict[Path, str]:
    """Output base names: file stem, disambiguated with the extension / a counter on collisions."""
    stems: dict[str, int] = {}
    for f in files:
        stems[f.stem] = stems.get(f.stem, 0) + 1
    names, used = {}, set()
    for f in files:
        name = f.stem if stems[f.stem] == 1 else f"{f.stem}_{f.suffix.lstrip('.').lower()}"
        base, i = name, 2
        while name in used:
            name, i = f"{base}_{i}", i + 1
        used.add(name)
        names[f] = name
    return names
