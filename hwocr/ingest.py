"""Turn input files (PDFs, images, multi-page TIFFs) into a stream of page images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageOps, ImageSequence

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif", ".heic", ".heif"}
PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = IMAGE_EXTS | PDF_EXTS

try:  # optional: iPhone photos
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass


@dataclass
class Page:
    source: Path
    number: int  # 1-based
    image: Image.Image


def discover_inputs(paths: list[str | Path]) -> list[Path]:
    """Expand files and folders (recursively) into a sorted list of supported documents."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(f for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS)
        elif p.is_file():
            if p.suffix.lower() not in SUPPORTED_EXTS:
                raise ValueError(f"Unsupported file type: {p}")
            found.append(p)
        else:
            raise FileNotFoundError(p)
    # de-duplicate while keeping order
    seen, unique = set(), []
    for f in found:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def parse_page_range(spec: str | None, total: int) -> list[int]:
    """'1-3,5' -> [1, 2, 3, 5] (1-based, clipped to total). None -> all pages."""
    if not spec:
        return list(range(1, total + 1))
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a or 1), int(b or total)
            pages.update(range(start, end + 1))
        elif part:
            pages.add(int(part))
    return sorted(p for p in pages if 1 <= p <= total)


def count_pages(path: Path) -> int:
    if path.suffix.lower() in PDF_EXTS:
        import pymupdf

        with pymupdf.open(path) as doc:
            return doc.page_count
    with Image.open(path) as im:
        return getattr(im, "n_frames", 1)


def iter_pages(path: Path, dpi: int = 200, pages: str | None = None) -> Iterator[Page]:
    """Yield RGB page images for a PDF or image file."""
    path = Path(path)
    wanted = parse_page_range(pages, count_pages(path))
    if path.suffix.lower() in PDF_EXTS:
        import pymupdf

        with pymupdf.open(path) as doc:
            for n in wanted:
                pix = doc[n - 1].get_pixmap(dpi=dpi, alpha=False, colorspace=pymupdf.csRGB)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                yield Page(path, n, img)
        return

    with Image.open(path) as im:
        for n, frame in enumerate(ImageSequence.Iterator(im), start=1):
            if n not in wanted:
                continue
            # Respect phone-camera EXIF orientation, flatten transparency onto white.
            frame = ImageOps.exif_transpose(frame.copy())
            if frame.mode in ("RGBA", "LA", "P"):
                frame = frame.convert("RGBA")
                bg = Image.new("RGB", frame.size, "white")
                bg.paste(frame, mask=frame.split()[-1])
                frame = bg
            yield Page(path, n, frame.convert("RGB"))
