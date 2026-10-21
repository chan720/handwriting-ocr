from hwocr.postprocess import (clean_output, collapse_repetition, is_degenerate, length_ratio_ok, merge_strips,
                               to_paragraphs)


def test_clean_strips_think_fences_and_preamble():
    raw = "<think>let me look</think>\nHere is the transcription of the page:\n```\nline one\nline two\n\n\n\nnext para\n```"
    assert clean_output(raw) == "line one\nline two\n\nnext para"


def test_clean_handles_dangling_think_close():
    assert clean_output("reasoning...</think>\nHello world") == "Hello world"


def test_clean_keeps_normal_text_starting_with_here():
    assert clean_output("Here is my garden.\nIt is small.") == "Here is my garden.\nIt is small."


def test_degenerate_line_loop():
    text = "Dear John,\n" + "and then we went home\n" * 12
    assert is_degenerate(text)
    assert not is_degenerate("a\nb\nc\nd\n\ne\nf")
    fixed = collapse_repetition(text)
    assert fixed.count("and then we went home") == 2


def test_degenerate_phrase_loop_in_one_line():
    assert is_degenerate("the cat sat on the big red mat " * 20)


def test_merge_strips_removes_overlap_and_keeps_longer_reading():
    a = "line one\nline two\nthe third line is quite long"
    b = "the thlrd line is quite lonq.\nline four\nline five"  # garbled where the strip edge cut it
    assert merge_strips(a, b) == "line one\nline two\nthe thlrd line is quite lonq.\nline four\nline five"


def test_merge_strips_does_not_merge_similar_short_lines():
    assert merge_strips("top half", "bottom half") == "top half\nbottom half"
    assert merge_strips("p1 l1\np1 l2", "p2 l1\np2 l2") == "p1 l1\np1 l2\np2 l1\np2 l2"


def test_merge_strips_without_overlap_concatenates():
    assert merge_strips("alpha\nbeta", "gamma\ndelta") == "alpha\nbeta\ngamma\ndelta"


def test_merge_strips_preserves_paragraph_breaks():
    a = "p1 l1\np1 l2\n\np2 l1"
    b = "p2 l1\np2 l2\n\np3 l1"
    assert to_paragraphs(merge_strips(a, b)) == [["p1 l1", "p1 l2"], ["p2 l1", "p2 l2"], ["p3 l1"]]


def test_to_paragraphs():
    assert to_paragraphs("a\nb\n\n  \nc") == [["a", "b"], ["c"]]


def test_length_ratio_guard():
    assert length_ratio_ok("hello world", "hello word")
    assert not length_ratio_ok("hello world " * 10, "hi")
