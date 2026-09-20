import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus import Chunk, Kind, load_chunks  # noqa: E402
from retrieval import (  # noqa: E402
    QUERY_PREFIX, Hit, Index, Retriever, apply_query_prefix, build_query,
    render_hits,
)

CORPUS_FILE = Path(os.environ.get("HIPKB_DATA", "data")) / "corpus" / "chunks.jsonl"
needs_corpus = pytest.mark.skipif(
    not CORPUS_FILE.is_file(), reason="corpus not built; run `python corpus.py`"
)


class StubEmbedder:
    """Hash-derived unit vectors. Deterministic, no download, real geometry."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.calls: list[bool] = []      # records is_query for each batch
        self.texts: list[str] = []

    def encode(self, texts, is_query: bool = False) -> np.ndarray:
        # A stub still owes the Embedder contract: prefix queries, not passages.
        self.calls.append(is_query)
        texts = apply_query_prefix(texts) if is_query else list(texts)
        self.texts.extend(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(self.dim).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out


def _chunk(i: int, kind: Kind, text: str = "") -> Chunk:
    return Chunk(
        chunk_id=f"doc{i}#0", doc_id=f"doc{i}", kind=kind,
        text=text or f"chunk body {i}", source="repo",
        path=f"docs/f{i}.rst", section=f"Section {i}",
    )


def _toy_retriever(n_ref: int = 40, n_ex: int = 40) -> tuple[Retriever, StubEmbedder]:
    emb = StubEmbedder()
    chunks = ([_chunk(i, Kind.REFERENCE) for i in range(n_ref)]
              + [_chunk(1000 + i, Kind.EXAMPLE) for i in range(n_ex)])
    return Retriever.build(emb, chunks), emb

def test_index_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="chunks but"):
        Index([_chunk(0, Kind.REFERENCE)], np.zeros((3, 8), dtype=np.float32))


def test_search_returns_scores_in_descending_order():
    emb = StubEmbedder()
    chunks = [_chunk(i, Kind.REFERENCE) for i in range(50)]
    idx = Index(chunks, emb.encode([c.text for c in chunks]))
    hits = idx.search(emb.encode(["some query"])[0], k=10)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), scores


def test_search_finds_the_exact_match_first():
    """A query identical to a passage must rank that passage top. This is what
    catches a normalisation or transpose bug -- the kind that still returns
    plausible-looking results."""
    emb = StubEmbedder()
    chunks = [_chunk(i, Kind.REFERENCE, text=f"unique body {i}") for i in range(50)]
    idx = Index(chunks, emb.encode([c.text for c in chunks]))
    target = chunks[17]
    hits = idx.search(emb.encode([target.text])[0], k=1)
    assert hits[0].chunk.chunk_id == target.chunk_id
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)


def test_search_caps_at_collection_size():
    emb = StubEmbedder()
    chunks = [_chunk(i, Kind.REFERENCE) for i in range(3)]
    idx = Index(chunks, emb.encode([c.text for c in chunks]))
    assert len(idx.search(emb.encode(["q"])[0], k=99)) == 3


def test_search_handles_empty_index_and_zero_k():
    emb = StubEmbedder()
    empty = Index([], np.zeros((0, 64), dtype=np.float32))
    assert empty.search(emb.encode(["q"])[0], k=5) == []
    chunks = [_chunk(0, Kind.REFERENCE)]
    idx = Index(chunks, emb.encode([chunks[0].text]))
    assert idx.search(emb.encode(["q"])[0], k=0) == []

def test_default_budget_is_seven_reference_plus_three_examples():
    r, _ = _toy_retriever()
    hits = r.retrieve("a query")
    assert len(hits) == 10, "must match the paper's top-10 token budget"
    assert sum(1 for h in hits if h.chunk.kind is Kind.REFERENCE) == 7
    assert sum(1 for h in hits if h.chunk.kind is Kind.EXAMPLE) == 3


def test_reference_chunks_come_first():
    """Prompt order affects what the model attends to. Reference is the half the
    paper found reliably helpful, so it leads."""
    r, _ = _toy_retriever()
    kinds = [h.chunk.kind for h in r.retrieve("a query")]
    assert kinds == [Kind.REFERENCE] * 7 + [Kind.EXAMPLE] * 3


def test_budget_is_overridable():
    r, _ = _toy_retriever()
    hits = r.retrieve("q", k_reference=2, k_examples=5)
    assert sum(1 for h in hits if h.chunk.kind is Kind.REFERENCE) == 2
    assert sum(1 for h in hits if h.chunk.kind is Kind.EXAMPLE) == 5


def test_build_refuses_a_single_kind_corpus():
    """A corpus missing one kind would silently make the two-index design
    collapse back into the paper's single index."""
    emb = StubEmbedder()
    with pytest.raises(ValueError, match="Need both kinds"):
        Retriever.build(emb, [_chunk(i, Kind.REFERENCE) for i in range(5)])


def test_retrieve_without_an_embedder_fails_loudly():
    r, _ = _toy_retriever()
    r.embedder = None
    with pytest.raises(ValueError, match="no embedder"):
        r.retrieve("q")


def test_prefix_applied_to_queries_only():
    """BGE is trained asymmetrically. Prefixing passages too would degrade
    retrieval just as much as omitting it from queries."""
    r, emb = _toy_retriever()
    emb.calls.clear()
    emb.texts.clear()
    r.retrieve("find me a reduction kernel")
    assert emb.calls == [True], "retrieve should embed exactly one query batch"
    assert emb.texts[0].startswith(QUERY_PREFIX)


def test_apply_query_prefix_is_directly_testable():
    """The whole reason this is a module-level function: the detail that most
    affects retrieval quality should not need a 1.3 GB download to verify."""
    out = apply_query_prefix(["reduction kernel", "matmul"])
    assert out == [QUERY_PREFIX + "reduction kernel", QUERY_PREFIX + "matmul"]
    assert apply_query_prefix([]) == []


def test_passages_embedded_without_prefix_at_build_time():
    emb = StubEmbedder()
    chunks = [_chunk(i, Kind.REFERENCE) for i in range(5)] + \
             [_chunk(9, Kind.EXAMPLE)]
    Retriever.build(emb, chunks)
    assert all(c is False for c in emb.calls), emb.calls
    assert not any(t.startswith(QUERY_PREFIX) for t in emb.texts)


def test_short_query_untouched():
    assert build_query("  short query  ") == "short query"


def test_long_query_truncated_at_a_line_boundary():
    """bge-large caps at 512 tokens. Cutting mid-line would hand the model a
    fragment of a Python statement as its search key."""
    text = "\n".join(f"line {i} with some content" for i in range(200))
    q = build_query(text)
    assert len(q) < len(text)
    assert not q.endswith("line")          # cut cleanly, not mid-token
    assert q.splitlines()[0] == "line 0 with some content"


def test_truncation_keeps_the_head_not_the_tail():
    """The head carries the class name and forward body; the tail is usually
    shape constants and get_inputs boilerplate."""
    head = "class Model(nn.Module):\n    def forward(self, x):\n        return x\n"
    tail = "\n".join(f"filler line {i}" for i in range(300))
    q = build_query(head + tail)
    assert "class Model" in q

def test_random_matches_the_treatment_shape():
    """The control must differ from `rag` in relevance and nothing else --
    same count, same collections, same split."""
    r, _ = _toy_retriever()
    real = r.retrieve("q")
    rand = r.retrieve_random("L1_P19")
    assert len(rand) == len(real)
    for kind in (Kind.REFERENCE, Kind.EXAMPLE):
        assert (sum(1 for h in rand if h.chunk.kind is kind)
                == sum(1 for h in real if h.chunk.kind is kind))


def test_random_is_reproducible_for_a_task():
    r, _ = _toy_retriever()
    a = [h.chunk.chunk_id for h in r.retrieve_random("L1_P19")]
    b = [h.chunk.chunk_id for h in r.retrieve_random("L1_P19")]
    assert a == b, "a rerun must retrieve the same irrelevant chunks"


def test_random_differs_across_tasks():
    r, _ = _toy_retriever()
    a = [h.chunk.chunk_id for h in r.retrieve_random("L1_P19")]
    b = [h.chunk.chunk_id for h in r.retrieve_random("L2_P44")]
    assert a != b


def test_random_needs_no_embedder():
    """The control arm must be runnable without loading a 1.3 GB model."""
    r, _ = _toy_retriever()
    r.embedder = None
    assert len(r.retrieve_random("L1_P19")) == 10


def test_random_draws_without_replacement():
    r, _ = _toy_retriever()
    ids = [h.chunk.chunk_id for h in r.retrieve_random("L1_P19")]
    assert len(ids) == len(set(ids))

def test_save_load_round_trip(tmp_path):
    r, emb = _toy_retriever()
    r.save(tmp_path)
    back = Retriever.load(tmp_path, emb)
    assert len(back.reference) == len(r.reference)
    assert back.reference.chunks[0] == r.reference.chunks[0]
    assert np.allclose(back.reference.vectors, r.reference.vectors)


def test_loaded_index_returns_identical_results(tmp_path):
    r, emb = _toy_retriever()
    before = [h.chunk.chunk_id for h in r.retrieve("a query")]
    r.save(tmp_path)
    after = [h.chunk.chunk_id for h in Retriever.load(tmp_path, emb).retrieve("a query")]
    assert before == after


def test_missing_index_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python retrieval.py"):
        Retriever.load(tmp_path / "nothing")


def test_rendered_hit_names_its_source():
    """A bare fragment is not usable context. The header is what makes it one."""
    h = Hit(_chunk(3, Kind.REFERENCE, text="the flags parameter must be zero"), 0.9)
    out = h.render()
    assert "docs/f3.rst" in out and "Section 3" in out
    assert "the flags parameter must be zero" in out


def test_render_hits_returns_plain_strings():
    r, _ = _toy_retriever()
    rendered = render_hits(r.retrieve("q"))
    assert len(rendered) == 10
    assert all(isinstance(s, str) and s for s in rendered)

@needs_corpus
def test_builds_over_the_real_corpus():
    r = Retriever.build(StubEmbedder(), load_chunks(CORPUS_FILE))
    assert len(r.reference) > 100 and len(r.examples) > 100
    assert len(r.retrieve("sum reduction over a float array")) == 10


@needs_corpus
def test_real_corpus_random_control_is_well_formed():
    r = Retriever.build(StubEmbedder(), load_chunks(CORPUS_FILE))
    hits = r.retrieve_random("L1_P19")
    assert len(hits) == 10
    assert all(h.chunk.text.strip() for h in hits)