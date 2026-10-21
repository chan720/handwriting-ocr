"""Accuracy measurement against hand-typed ground truth.

Metrics
- CER  (character error rate)  = edit distance / reference characters.   Char accuracy = 1 - CER.
- WER  (word error rate)       = word-level edit distance / reference words.
- Line F1                      = how well the output's line breaks match the page (in-order fuzzy line match).
- Paragraph count              = reference vs predicted number of paragraphs.

"80% accuracy" is ambiguous; this reports both character and word accuracy so you can see which you mean.
Word accuracy is the stricter one: a single wrong letter makes the whole word wrong.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from .postprocess import to_paragraphs

_MARKERS = re.compile(r"\[(illegible|\?)\]", re.IGNORECASE)


def normalize(text: str, lowercase: bool = False, strip_punct: bool = False) -> str:
    text = _MARKERS.sub(" ", text)
    if lowercase:
        text = text.lower()
    if strip_punct:
        text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _lines(text: str) -> list[str]:
    return [normalize(ln) for ln in text.split("\n") if normalize(ln)]


def line_f1(ref: str, hyp: str, threshold: float = 80.0, window: int = 6) -> float:
    r, h = _lines(ref), _lines(hyp)
    if not r and not h:
        return 1.0
    if not r or not h:
        return 0.0
    matched, j = 0, 0
    for line in r:
        best, best_k = 0.0, -1
        for k in range(j, min(len(h), j + window)):
            s = fuzz.ratio(line, h[k])
            if s > best:
                best, best_k = s, k
        if best >= threshold:
            matched += 1
            j = best_k + 1
    p, rc = matched / len(h), matched / len(r)
    return 0.0 if p + rc == 0 else 2 * p * rc / (p + rc)


@dataclass
class Score:
    name: str
    ref_chars: int
    char_edits: int
    ref_words: int
    word_edits: int
    cer: float
    wer: float
    line_f1: float
    ref_paragraphs: int
    pred_paragraphs: int

    @property
    def char_accuracy(self) -> float:
        return max(0.0, 1 - self.cer)

    @property
    def word_accuracy(self) -> float:
        return max(0.0, 1 - self.wer)


def score_texts(name: str, ref: str, hyp: str, lowercase: bool = False, strip_punct: bool = False) -> Score:
    r, h = normalize(ref, lowercase, strip_punct), normalize(hyp, lowercase, strip_punct)
    ce = Levenshtein.distance(r, h)
    rw, hw = r.split(), h.split()
    we = Levenshtein.distance(rw, hw)
    return Score(
        name=name, ref_chars=len(r), char_edits=ce, ref_words=len(rw), word_edits=we,
        cer=ce / max(1, len(r)), wer=we / max(1, len(rw)),
        line_f1=line_f1(ref, hyp),
        ref_paragraphs=len(to_paragraphs(ref)), pred_paragraphs=len(to_paragraphs(hyp)),
    )


def evaluate_dirs(pred_dir: str | Path, gt_dir: str | Path, **norm) -> tuple[list[Score], dict]:
    """Score every <name>.txt in gt_dir against <name>.txt in pred_dir."""
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    scores, missing = [], []
    for gt in sorted(gt_dir.glob("*.txt")):
        pred = pred_dir / gt.name
        if not pred.exists():
            missing.append(gt.stem)
            continue
        scores.append(score_texts(gt.stem, gt.read_text(encoding="utf-8"), pred.read_text(encoding="utf-8"), **norm))
    return scores, summarize(scores, missing)


def summarize(scores: list[Score], missing: list[str] | None = None) -> dict:
    if not scores:
        return {"documents": 0, "missing": missing or []}
    chars = sum(s.ref_chars for s in scores)
    words = sum(s.ref_words for s in scores)
    cer = sum(s.char_edits for s in scores) / max(1, chars)
    wer = sum(s.word_edits for s in scores) / max(1, words)
    return {
        "documents": len(scores),
        "cer": cer, "wer": wer,
        "char_accuracy": max(0.0, 1 - cer), "word_accuracy": max(0.0, 1 - wer),
        "line_f1": sum(s.line_f1 for s in scores) / len(scores),
        "paragraph_count_exact": sum(s.ref_paragraphs == s.pred_paragraphs for s in scores) / len(scores),
        "missing": missing or [],
    }


def format_report(scores: list[Score], summary: dict) -> str:
    rows = ["| document | char acc | word acc | CER | WER | line F1 | paragraphs (ref/pred) |",
            "|---|---|---|---|---|---|---|"]
    for s in scores:
        rows.append(f"| {s.name} | {s.char_accuracy:.1%} | {s.word_accuracy:.1%} | {s.cer:.3f} | {s.wer:.3f} "
                    f"| {s.line_f1:.2f} | {s.ref_paragraphs}/{s.pred_paragraphs} |")
    if summary.get("documents"):
        rows.append(f"| **OVERALL** | **{summary['char_accuracy']:.1%}** | **{summary['word_accuracy']:.1%}** "
                    f"| {summary['cer']:.3f} | {summary['wer']:.3f} | {summary['line_f1']:.2f} "
                    f"| {summary['paragraph_count_exact']:.0%} exact |")
    if summary.get("missing"):
        rows.append(f"\nMissing predictions for: {', '.join(summary['missing'])}")
    return "\n".join(rows)


def score_dicts(scores: list[Score]) -> list[dict]:
    return [{**asdict(s), "char_accuracy": s.char_accuracy, "word_accuracy": s.word_accuracy} for s in scores]
