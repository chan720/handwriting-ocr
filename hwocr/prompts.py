"""Prompts. These matter a lot: a strict transcription prompt is what stops VLMs from summarising,
'fixing' spelling, or flattening the page into one paragraph."""

from __future__ import annotations

from .config import PromptConfig

SYSTEM_PROMPT = (
    "You are a meticulous transcription engine for handwritten documents. "
    "You copy text exactly as written. You never summarise, explain, translate or correct."
)

_RULES = """Rules:
1. Output ONLY the transcription. No introduction, no commentary, no code fences.
2. Follow the natural reading order of the page ({direction}). Margin notes go after the main text block they sit beside.
3. Preserve line breaks: every handwritten line becomes exactly one line in your output.
4. Separate paragraphs with exactly ONE blank line. A new paragraph starts where the writer leaves vertical space, indents a line, or begins a new block (heading, date, list, address, signature).
5. Copy spelling, grammar, punctuation, capitalisation and numbers exactly as written - do not correct or modernise anything.
6. Text that is crossed out: leave it out. Words inserted above the line with a caret: place them where they are inserted.
7. If a word cannot be read, write [illegible]. If you are unsure of a word, give your best reading followed by [?]. Never invent text that is not on the page.
8. Keep list bullets/numbers. Write table rows on one line with " | " between cells."""


def _direction(cfg: PromptConfig) -> str:
    return "top to bottom, right to left" if cfg.right_to_left else "top to bottom, left to right"


def _extras(cfg: PromptConfig) -> str:
    parts = []
    if cfg.language:
        parts.append(f"The document is written in {cfg.language}. Transcribe it in the original script; do not translate.")
    if cfg.context:
        parts.append(f"Context: {cfg.context}.")
    return ("\n" + "\n".join(parts)) if parts else ""


def transcribe_prompt(cfg: PromptConfig, is_strip: bool = False) -> str:
    head = "Transcribe all handwritten (and any printed) text in this image."
    if is_strip:
        head += (
            " This image is one horizontal strip of a taller page, so a line at the very top or bottom edge"
            " may be cut off - transcribe whatever part of it is readable."
        )
    return f"{head}{_extras(cfg)}\n\n{_RULES.format(direction=_direction(cfg))}"


def refine_prompt(cfg: PromptConfig, draft: str, is_strip: bool = False) -> str:
    scope = "this strip of a page" if is_strip else "this page"
    return (
        f"Below is a draft transcription of {scope}, produced by an automatic reader. It may contain misread words, "
        "missing or duplicated lines, and wrong line or paragraph breaks.\n"
        "Compare it against the image word by word and output the corrected transcription. "
        "If the draft is already correct, output it unchanged."
        f"{_extras(cfg)}\n\n{_RULES.format(direction=_direction(cfg))}\n\n"
        f"<draft>\n{draft}\n</draft>"
    )
