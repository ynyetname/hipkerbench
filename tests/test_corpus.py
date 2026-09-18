import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus import (  # noqa: E402
    Chunk, Document, Kind, _should_skip, chunk_document, load_chunks,
    recursive_split, split_by_headers, split_code,
)

CORPUS_FILE = Path(os.environ.get("HIPKB_DATA", "data")) / "corpus" / "chunks.jsonl"
needs_corpus = pytest.mark.skipif(
    not CORPUS_FILE.is_file(), reason="corpus not built; run `python corpus.py`"
)

RST = """Memory management
=================

Intro paragraph about memory.

Coherency
---------

How coherency works.

Fine-grained
~~~~~~~~~~~~

Details on fine-grained memory.

Streams
-------

Back up a level, to a sibling of Coherency.
"""

def test_rst_builds_a_section_trail():
    """A retrieved fragment is useless without knowing where it came from.
    'the flags parameter must be zero' means nothing on its own."""
    sections = dict(split_by_headers(RST))
    assert "Memory management > Coherency > Fine-grained" in sections
    assert "fine-grained" in sections["Memory management > Coherency > Fine-grained"]

def test_rst_level_is_decided_by_order_of_first_appearance():
    """RST has no fixed hierarchy -- '=' is not inherently above '-'. The level
    is whatever order the characters first appear in that file."""
    trails = [s for s, _ in split_by_headers(RST)]
    assert "Memory management > Streams" in trails, trails
    assert "Memory management > Coherency > Streams" not in trails

def test_rst_underline_must_be_long_enough():
    """A short run of dashes is a list bullet or a table rule, not a header."""
    text = "Some prose here\n---\nmore prose\n"
    sections = split_by_headers(text)
    assert all(s == "" for s, _ in sections), sections

def test_rst_ignores_underline_with_no_title():
    text = "\n=====\n\nJust a horizontal rule above.\n"
    assert all(s == "" for s, _ in split_by_headers(text))

MD = """# Getting started

Intro.

## Building

How to build.

### Requirements

You need a compiler.

## Running

Back to level two.
"""

def test_markdown_trail_and_depth():
    sections = dict(split_by_headers(MD, is_markdown=True))
    assert "Getting started > Building > Requirements" in sections
    assert "Getting started > Running" in sections

def test_markdown_not_parsed_as_rst_by_default():
    """Passing Markdown through the RST path would produce one giant section."""
    assert len(split_by_headers(MD, is_markdown=True)) > 1

KERNEL_SRC = """#include <hip/hip_runtime.h>

__global__ void add(const float* a, const float* b, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        o[i] = a[i] + b[i];
    }
}

__global__ void scale(float* v, float k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        v[i] *= k;
    }
}
"""

def test_code_splits_at_function_boundaries():
    parts = split_code(KERNEL_SRC)
    assert len(parts) >= 2, parts

def test_code_never_cuts_mid_function():
    """A kernel cut in half is worse than useless as retrieved context -- it
    teaches syntax that will not compile."""
    for _name, body in split_code(KERNEL_SRC):
        assert body.count("{") == body.count("}"), f"unbalanced braces:\n{body}"

def test_code_keeps_each_kernel_whole():
    bodies = [b for _n, b in split_code(KERNEL_SRC)]
    joined = "\n".join(bodies)
    assert "o[i] = a[i] + b[i];" in joined
    assert "v[i] *= k;" in joined

def test_code_split_handles_empty_and_trivial_input():
    assert split_code("") == []
    assert split_code("   \n\n  ") == []

def test_short_text_is_one_chunk():
    assert recursive_split("short", chunk_size=1000) == ["short"]

def test_respects_chunk_size():
    text = "\n\n".join(f"Paragraph number {i}. " * 12 for i in range(40))
    for c in recursive_split(text, chunk_size=500, overlap=50):
        assert len(c) <= 500 + 50, len(c)

def test_falls_through_separators_when_no_paragraph_breaks():
    """One long unbroken line must still be split, not returned oversized."""
    text = "word " * 2000
    chunks = recursive_split(text, chunk_size=300, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)

def test_no_empty_chunks():
    text = "\n\n\n\n".join(["alpha", "", "   ", "beta"])
    assert all(c.strip() for c in recursive_split(text, chunk_size=50))

def test_overlap_carries_context_across_a_cut():
    text = "\n\n".join(f"Section {i} body text here." for i in range(30))
    with_overlap = recursive_split(text, chunk_size=200, overlap=60)
    without = recursive_split(text, chunk_size=200, overlap=0)
    assert sum(len(c) for c in with_overlap) > sum(len(c) for c in without)

def test_skips_sphinx_and_build_machinery():
    assert _should_skip(Path("docs/sphinx/conf.py"))
    assert _should_skip(Path("docs/doxygen/Doxyfile"))
    assert _should_skip(Path("HIP-Basic/saxpy/CMakeLists.txt"))
    assert not _should_skip(Path("docs/how-to/memory.rst"))
    assert not _should_skip(Path("HIP-Basic/saxpy/main.hip"))

def test_example_documents_use_the_code_splitter():
    doc = Document("d", "repo", "HIP-Basic/saxpy/main.hip", Kind.EXAMPLE, KERNEL_SRC)
    for c in chunk_document(doc):
        assert c.text.count("{") == c.text.count("}")

def test_reference_documents_carry_a_section_trail():
    doc = Document("d", "repo", "docs/mem.rst", Kind.REFERENCE, RST)
    assert any(c.section for c in chunk_document(doc))

def test_chunk_ids_are_unique_within_a_document():
    doc = Document("d", "repo", "docs/mem.rst", Kind.REFERENCE, RST)
    ids = [c.chunk_id for c in chunk_document(doc)]
    assert len(ids) == len(set(ids))

def test_chunks_keep_their_kind():
    ref = Document("a", "r", "docs/x.rst", Kind.REFERENCE, RST)
    ex = Document("b", "r", "HIP-Basic/x.hip", Kind.EXAMPLE, KERNEL_SRC)
    assert all(c.kind is Kind.REFERENCE for c in chunk_document(ref))
    assert all(c.kind is Kind.EXAMPLE for c in chunk_document(ex))

def test_chunk_json_round_trip():
    c = Chunk("id#0", "id", Kind.REFERENCE, "body", "repo", "docs/x.rst",
              section="A > B", metadata={"part": 2})
    back = Chunk.from_json(c.to_json())
    assert back == c
    assert back.kind is Kind.REFERENCE, "kind must survive as an enum, not a string"

def test_missing_corpus_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python corpus.py"):
        load_chunks(tmp_path / "nope.jsonl")

@needs_corpus
def test_real_corpus_has_both_kinds():
    chunks = load_chunks(CORPUS_FILE)
    kinds = {c.kind for c in chunks}
    assert kinds == {Kind.REFERENCE, Kind.EXAMPLE}, kinds

@needs_corpus
def test_real_chunks_respect_the_size_cap():
    from config import RETRIEVAL
    oversized = [c for c in load_chunks(CORPUS_FILE)
                 if len(c.text) > RETRIEVAL.chunk_size + RETRIEVAL.chunk_overlap]
    assert not oversized, f"{len(oversized)} chunks exceed the cap"

@needs_corpus
def test_real_corpus_is_big_enough_to_retrieve_from():
    from config import RETRIEVAL
    chunks = load_chunks(CORPUS_FILE)
    for kind, k in ((Kind.REFERENCE, RETRIEVAL.top_k_reference),
                    (Kind.EXAMPLE, RETRIEVAL.top_k_examples)):
        n = sum(1 for c in chunks if c.kind is kind)
        assert n > k * 10, f"only {n} {kind.value} chunks for top-{k} retrieval"