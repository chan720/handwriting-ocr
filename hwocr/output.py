"""Write results as plain text, Markdown and structured JSON (pages -> paragraphs -> lines)."""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .pipeline import DocumentResult


def to_txt(doc: DocumentResult, cfg: Config) -> str:
    return cfg.output.page_separator.join(p.text for p in doc.pages if p.text).strip() + "\n"


def to_markdown(doc: DocumentResult) -> str:
    out = [f"# {Path(doc.source).name}", ""]
    for p in doc.pages:
        if len(doc.pages) > 1:
            out += [f"## Page {p.page}", ""]
        for para in p.paragraphs:
            # Two trailing spaces = Markdown hard line break, so handwritten line breaks survive rendering.
            out.append("  \n".join(para))
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def to_json(doc: DocumentResult) -> dict:
    return {
        "source": doc.source,
        "model": doc.model,
        "backend": doc.backend,
        "page_count": len(doc.pages),
        "pages": [
            {
                "page": p.page,
                "text": p.text,
                "paragraphs": [{"index": i, "lines": lines, "text": "\n".join(lines)}
                               for i, lines in enumerate(p.paragraphs)],
                "strategy": p.strategy,
                "flags": p.flags,
                "seconds": p.seconds,
                "preprocess": p.preprocess,
            }
            for p in doc.pages
        ],
    }


def write_outputs(doc: DocumentResult, out_dir: Path, cfg: Config) -> list[Path]:
    written = []
    fmts = set(cfg.output.formats)
    if "txt" in fmts:
        f = out_dir / f"{doc.name}.txt"
        f.write_text(to_txt(doc, cfg), encoding="utf-8")
        written.append(f)
    if "md" in fmts:
        f = out_dir / f"{doc.name}.md"
        f.write_text(to_markdown(doc), encoding="utf-8")
        written.append(f)
    if "json" in fmts:
        f = out_dir / f"{doc.name}.json"
        f.write_text(json.dumps(to_json(doc), ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(f)
    return written
