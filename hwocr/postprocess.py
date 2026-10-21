"""Cleaning model output, catching degenerate generations, merging strips, and structuring text."""

from __future__ import annotations

import re
from collections import Counter

from rapidfuzz import fuzz

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)
_PREAMBLE_RE = re.compile(
    r"^\s*(sure[,!.]?\s*)?(here\s+is|here's|below\s+is)\b[^\n]*\b(transcri\w*|text)\b[^\n]*:\s*\n",
    re.IGNORECASE,
)
_DRAFT_TAG_RE = re.compile(r"</?draft>", re.IGNORECASE)


def clean_output(text: str) -> str:
    """Strip reasoning blocks, code fences, chatty preambles; normalise whitespace and paragraph gaps."""
    text = _THINK_RE.sub("", text or "")
    if "</think>" in text:  # opening tag was in the prompt template, only the close was generated
        text = text.split("</think>", 1)[1]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _PREAMBLE_RE.sub("", text, count=1)
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1)
    text = _DRAFT_TAG_RE.sub("", text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n[ \t]*\n(\s*\n)+", "\n\n", text)  # 2+ blank lines -> exactly one
    return text.strip("\n").strip()


def is_degenerate(text: str, max_line_repeats: int = 4) -> bool:
    """Detect the classic VLM failure: the same line or phrase repeated until max_tokens."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        run, longest = 1, 1
        for a, b in zip(lines, lines[1:]):
            run = run + 1 if a == b else 1
            longest = max(longest, run)
        if longest > max_line_repeats:
            return True
        counts = Counter(lines)
        common, n = counts.most_common(1)[0]
        if n > max_line_repeats * 2 and n / len(lines) > 0.3 and len(common) > 3:
            return True
    # phrase-level loop inside one long line: an 8-word window repeated many times
    words = text.split()
    if len(words) >= 80:
        grams = Counter(tuple(words[i:i + 8]) for i in range(len(words) - 7))
        if grams and grams.most_common(1)[0][1] >= 6:
            return True
    return False


def collapse_repetition(text: str, max_line_repeats: int = 2) -> str:
    """Last-resort repair: drop consecutive duplicate lines beyond max_line_repeats."""
    out: list[str] = []
    run = 0
    for ln in text.split("\n"):
        if out and ln.strip() and ln.strip() == out[-1].strip():
            run += 1
            if run >= max_line_repeats:
                continue
        else:
            run = 0
        out.append(ln)
    return "\n".join(out)


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def merge_strips(a: str, b: str, max_overlap: int = 6, threshold: float = 80.0) -> str:
    """Join two strip transcriptions, removing lines duplicated in the overlap region.

    Finds the largest k such that the last k non-empty lines of `a` fuzzily match the first k
    non-empty lines of `b`. Short lines must match exactly (one letter changes a lot of "similarity"
    on short strings). For each matched pair the longer reading is kept: the strip edge slices
    through a line, so one side usually has a garbled or shortened version.
    """
    if not a.strip():
        return b
    if not b.strip():
        return a
    a_lines, b_lines = a.split("\n"), b.split("\n")
    a_idx = [i for i, ln in enumerate(a_lines) if ln.strip()]
    b_idx = [i for i, ln in enumerate(b_lines) if ln.strip()]

    best_k = 0
    for k in range(min(max_overlap, len(a_idx), len(b_idx)), 0, -1):
        tail = [a_lines[i] for i in a_idx[-k:]]
        head = [b_lines[i] for i in b_idx[:k]]
        scores = []
        for x, y in zip(tail, head):
            nx, ny = _norm(x), _norm(y)
            if not nx or not ny:
                scores.append(0.0)
                continue
            if min(len(nx), len(ny)) < 12:
                scores.append(100.0 if nx == ny else 0.0)
            else:
                scores.append(fuzz.ratio(nx, ny))
        if min(scores) >= threshold:
            best_k = k
            break

    if best_k == 0:
        return a.rstrip("\n") + "\n" + b.lstrip("\n")

    keep_a = a_lines[: a_idx[-best_k]]
    overlap = []
    for i, j in zip(a_idx[-best_k:], b_idx[:best_k]):
        x, y = a_lines[i], b_lines[j]
        overlap.append(x if len(x.strip()) >= len(y.strip()) else y)
    rest_b = b_lines[b_idx[best_k - 1] + 1:]
    return "\n".join(keep_a + overlap + rest_b)


def to_paragraphs(text: str) -> list[list[str]]:
    """'line\\nline\\n\\nline' -> [['line', 'line'], ['line']]."""
    paragraphs = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if lines:
            paragraphs.append(lines)
    return paragraphs


def length_ratio_ok(draft: str, revised: str, low: float = 0.6, high: float = 1.6) -> bool:
    """Guard for the refine pass: reject a revision that drastically shrinks or grows the text."""
    d, r = len(draft.strip()), len(revised.strip())
    if d == 0:
        return r > 0
    return low <= r / d <= high
