import pymupdf
import pytest
from PIL import Image

from hwocr.config import PreprocessConfig
from hwocr.ingest import discover_inputs, iter_pages, parse_page_range
from hwocr.preprocess import _to_gray, estimate_skew, preprocess, split_into_strips


@pytest.mark.parametrize("angle", [-3.0, 0.0, 2.5])
def test_estimate_skew_recovers_rotation(page_image, angle):
    rotated = page_image.rotate(angle, expand=True, fillcolor="white")
    est = estimate_skew(_to_gray(rotated))
    assert abs(est + angle) <= 0.5  # estimate is the correction (opposite sign)


def test_preprocess_resizes_and_deskews(page_image):
    big = page_image.rotate(2, expand=True, fillcolor="white").resize((2480, 3508))
    out, info = preprocess(big, PreprocessConfig(max_side=2048))
    assert max(out.size) == 2048
    assert abs(info["deskew_angle"] + 2) <= 0.5


def test_preprocess_upscales_small_photo(page_image):
    out, _ = preprocess(page_image.resize((400, 566)), PreprocessConfig(min_side=1200, deskew=False))
    assert max(out.size) == 1200


def test_split_into_strips_cuts_in_whitespace(page_image):
    strips = split_into_strips(page_image, 2, 0.08)
    assert len(strips) == 2
    assert sum(s.height for s in strips) == page_image.height  # clean gap cut -> no overlap


def test_split_into_strips_overlaps_when_no_gap():
    solid = Image.new("RGB", (800, 1000), "black")  # ink everywhere: no clean gap anywhere
    strips = split_into_strips(solid, 2, 0.1)
    assert sum(s.height for s in strips) > 1000


def test_parse_page_range():
    assert parse_page_range("1-3,5,9", 6) == [1, 2, 3, 5]
    assert parse_page_range(None, 3) == [1, 2, 3]
    assert parse_page_range("4-", 6) == [4, 5, 6]


def test_ingest_pdf_and_multipage_tiff(tmp_path, page_image):
    pdf_path = tmp_path / "notes.pdf"
    page_image.save(tmp_path / "p.png")
    doc = pymupdf.open()
    for _ in range(3):
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, filename=str(tmp_path / "p.png"))
    doc.save(pdf_path)

    pages = list(iter_pages(pdf_path, dpi=150, pages="2-3"))
    assert [p.number for p in pages] == [2, 3]
    assert pages[0].image.mode == "RGB" and pages[0].image.width > 1000

    tif = tmp_path / "scan.tiff"
    page_image.save(tif, save_all=True, append_images=[page_image.convert("L")])
    assert [p.number for p in iter_pages(tif)] == [1, 2]

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "ignore.txt").write_text("x")
    page_image.save(sub / "photo.jpg")
    found = discover_inputs([tmp_path])
    assert {f.name for f in found} == {"notes.pdf", "scan.tiff", "photo.jpg", "p.png"}
