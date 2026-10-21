"""Command-line interface.

    python -m hwocr doctor                       # check the model server / model is ready
    python -m hwocr run INPUT... -o output       # transcribe PDFs / images / folders
    python -m hwocr eval --pred output --gt samples/ground_truth
    python -m hwocr bench --inputs samples/input --gt samples/ground_truth --models qwen3-vl:8b,qwen3.5:9b
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path

from .backends import BackendError, make_backend
from .config import load_config
from .evaluate import evaluate_dirs, format_report, score_dicts

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def _add_common(p: argparse.ArgumentParser):
    p.add_argument("-c", "--config", default=str(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else None,
                   help="YAML config file (default: ./config.yaml)")
    p.add_argument("--backend", dest="backend.type", help="ollama | openai | transformers")
    p.add_argument("--base-url", dest="backend.base_url")
    p.add_argument("--model", dest="backend.model", help="model name on the server (e.g. qwen3-vl:8b)")
    p.add_argument("--hf-model", dest="transformers.model_id", help="HF model id for --backend transformers")
    p.add_argument("--max-tokens", dest="backend.max_tokens", type=int)
    p.add_argument("--temperature", dest="backend.temperature", type=float,
                   help="0 = greedy (default, best for instruct models); ~0.6 for thinking models")
    p.add_argument("--gpu-layers", dest="backend.num_gpu", type=int,
                   help="Ollama: how many model layers to put on the GPU (0 = CPU only)")
    p.add_argument("--timeout", dest="backend.timeout", type=float,
                   help="seconds to wait for one page before giving up (default 900)")
    p.add_argument("--ctx", dest="backend.num_ctx", type=int,
                   help="Ollama context size (default 16384; 8192 saves ~1 GB memory on 8B models)")
    p.add_argument("--dpi", dest="ingest.pdf_dpi", type=int)
    p.add_argument("--tiling", dest="tiling.mode", choices=["off", "fallback", "always"])
    p.add_argument("--strips", dest="tiling.strips", type=int)
    p.add_argument("--refine", dest="refine.enabled", action="store_const", const=True,
                   help="second pass where the model proof-reads its own draft against the image")
    p.add_argument("--language", dest="prompt.language", help='e.g. "English", "Urdu"')
    p.add_argument("--rtl", dest="prompt.right_to_left", action="store_const", const=True,
                   help="right-to-left script (Arabic, Urdu, Hebrew, Persian)")
    p.add_argument("--context", dest="prompt.context", help='e.g. "doctor\'s prescription"')
    p.add_argument("--contrast", dest="preprocess.enhance_contrast", action="store_const", const=True,
                   help="boost faint pencil / faded ink")
    p.add_argument("--denoise", dest="preprocess.denoise", action="store_const", const=True)
    p.add_argument("--no-deskew", dest="preprocess.deskew", action="store_const", const=False)
    p.add_argument("--rotate", dest="preprocess.rotate", type=int, choices=[0, 90, 180, 270],
                   help="rotate pages clockwise before reading")
    p.add_argument("--workers", dest="runtime.workers", type=int)
    p.add_argument("--threads", dest="backend.num_thread", type=int,
                   help="Ollama CPU threads (default: all cores); lower = lighter load, slower")
    p.add_argument("--max-side", dest="preprocess.max_side", type=int,
                   help="shrink pages so the longest side is at most this many pixels (default 2048)")
    p.add_argument("--debug-images", dest="output.save_debug_images", action="store_const", const=True)
    p.add_argument("-v", "--verbose", action="store_true")


def _cfg_from_args(args):
    overrides = {k: v for k, v in vars(args).items() if "." in k}
    return load_config(args.config, overrides)


def cmd_doctor(args) -> int:
    cfg = _cfg_from_args(args)
    try:
        print("OK:", make_backend(cfg).check())
        return 0
    except BackendError as e:
        print("PROBLEM:", e)
        return 1


def cmd_run(args) -> int:
    from .pipeline import Pipeline

    cfg = _cfg_from_args(args)
    t0 = time.time()
    docs = Pipeline(cfg).run(args.inputs, args.output, pages=args.pages, overwrite=args.overwrite)
    n_pages = sum(len(d.pages) for d in docs)
    flagged = [(d.name, p.page, p.flags) for d in docs for p in d.pages
               if {"degenerate_output", "truncated"} & set(p.flags)]
    print(f"\nDone: {len(docs)} document(s), {n_pages} page(s) in {time.time() - t0:.0f}s -> {args.output}")
    for name, page, flags in flagged:
        print(f"  check manually: {name} page {page} ({', '.join(flags)})")
    return 0


def cmd_eval(args) -> int:
    scores, summary = evaluate_dirs(args.pred, args.gt, lowercase=args.ignore_case, strip_punct=args.ignore_punct)
    print(format_report(scores, summary))
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "documents": score_dicts(scores)}, indent=2))
    return 0 if scores else 1


def cmd_bench(args) -> int:
    from .pipeline import Pipeline

    base = _cfg_from_args(args)
    models = [m.strip() for m in args.models.split(",") if m.strip()] or [base.backend.model]
    out_root = Path(args.output)
    rows = []
    for model in models:
        cfg = copy.deepcopy(base)
        if cfg.backend.type in ("transformers", "hf"):
            cfg.transformers.model_id = model
        else:
            cfg.backend.model = model
        out_dir = out_root / re.sub(r"[^\w.-]+", "_", model)
        print(f"\n=== {model} ===")
        t0 = time.time()
        try:
            docs = Pipeline(cfg).run([args.inputs], out_dir, pages=args.pages, overwrite=args.overwrite)
        except BackendError as e:
            print(f"  skipped: {e}")
            continue
        pages = sum(len(d.pages) for d in docs) or 1
        page_secs = sum(p.seconds for d in docs for p in d.pages) / pages
        scores, summary = evaluate_dirs(out_dir, args.gt, lowercase=args.ignore_case, strip_punct=args.ignore_punct)
        print(format_report(scores, summary))
        if summary.get("documents"):
            rows.append({"model": model, **{k: summary[k] for k in
                         ("char_accuracy", "word_accuracy", "cer", "wer", "line_f1", "paragraph_count_exact")},
                         "sec_per_page": page_secs, "wall_seconds": time.time() - t0})

    if not rows:
        print("No results.")
        return 1
    rows.sort(key=lambda r: r["cer"])
    table = ["| model | char acc | word acc | line F1 | paragraphs exact | s/page |", "|---|---|---|---|---|---|"]
    for r in rows:
        table.append(f"| {r['model']} | {r['char_accuracy']:.1%} | {r['word_accuracy']:.1%} | {r['line_f1']:.2f} "
                     f"| {r['paragraph_count_exact']:.0%} | {r['sec_per_page']:.1f} |")
    report = "\n".join(table)
    print("\n=== Leaderboard (best first) ===\n" + report)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "leaderboard.md").write_text(report + "\n", encoding="utf-8")
    with open(out_root / "leaderboard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out_root / 'leaderboard.md'} and leaderboard.csv")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hwocr", description="Offline handwritten PDF/image -> text pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="check that the model backend is reachable and the model is available")
    _add_common(p)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("run", help="transcribe PDFs / images / folders")
    p.add_argument("inputs", nargs="+", help="files or folders")
    p.add_argument("-o", "--output", default="output")
    p.add_argument("--pages", help='page selection, e.g. "1-3,7"')
    p.add_argument("--overwrite", action="store_true", help="ignore cached page results")
    _add_common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("eval", help="score transcriptions against ground-truth .txt files")
    p.add_argument("--pred", required=True, help="folder with <name>.txt predictions")
    p.add_argument("--gt", required=True, help="folder with <name>.txt ground truth")
    p.add_argument("--ignore-case", action="store_true")
    p.add_argument("--ignore-punct", action="store_true")
    p.add_argument("--json", help="also write the scores to this JSON file")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("bench", help="run several models on a labelled set and rank them")
    p.add_argument("--inputs", required=True, help="folder of PDFs/images")
    p.add_argument("--gt", required=True, help="folder of <name>.txt ground truth")
    p.add_argument("--models", required=True, help="comma-separated model names")
    p.add_argument("-o", "--output", default="bench_output")
    p.add_argument("--pages")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--ignore-case", action="store_true")
    p.add_argument("--ignore-punct", action="store_true")
    _add_common(p)
    p.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    try:
        return args.func(args)
    except BackendError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
