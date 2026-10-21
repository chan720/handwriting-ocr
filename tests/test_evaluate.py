from hwocr.evaluate import evaluate_dirs, line_f1, score_texts


def test_perfect_match():
    s = score_texts("x", "Hello world\n\nNew para", "Hello world\n\nNew para")
    assert s.cer == 0 and s.wer == 0 and s.line_f1 == 1.0
    assert s.ref_paragraphs == s.pred_paragraphs == 2


def test_errors_counted():
    s = score_texts("x", "the quick brown fox", "the quack brown fox")
    assert s.char_edits == 1
    assert s.word_edits == 1
    assert round(s.word_accuracy, 2) == 0.75


def test_markers_ignored_and_whitespace_normalised():
    s = score_texts("x", "dear  john\nhow are you", "dear [illegible] john how are you [?]")
    assert s.cer == 0


def test_line_f1_penalises_merged_lines():
    ref = "line one here\nline two here\nline three here"
    assert line_f1(ref, ref) == 1.0
    assert line_f1(ref, "line one here line two here line three here") == 0.0


def test_evaluate_dirs(tmp_path):
    gt, pred = tmp_path / "gt", tmp_path / "pred"
    gt.mkdir(), pred.mkdir()
    (gt / "a.txt").write_text("abcd efgh")
    (pred / "a.txt").write_text("abcd efgx")
    (gt / "b.txt").write_text("missing one")
    scores, summary = evaluate_dirs(pred, gt)
    assert len(scores) == 1 and summary["missing"] == ["b"]
    assert abs(summary["cer"] - 1 / 9) < 1e-9
