from __future__ import annotations
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence
import numpy as np
 
from config import INDEX_DIR, RETRIEVAL
from corpus import Chunk, Kind, load_chunks

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MAX_QUERY_CHARS = 1400

class Embedder(Protocol):
    """Anything that turns text into vectors. A Protocol so tests can inject a
    stub and exercise the whole pipeline without downloading 1.3 GB."""
    
    def encode(self, texts: Sequence[str], is_query: bool=False) -> np.ndarray: ...
    
class BGEEmbedder:
    """The real thing: BAAI/bge-large-en-v1.5"""
    def __init__(self, model_name: str | None=None, device: str | None=None, batch_size: int=16):
        from sentence_transformers import SentenceTransformer
        self.model_name=model_name or RETRIEVAL.embed_model
        self.batch_size = batch_size
        self.model = SentenceTransformer(self.model_name, device=device)
        
    def encode(self, texts: Sequence[str], is_query: bool=False) -> np.ndarray:
        payload = [QUERY_PREFIX + t for t in texts] if is_query else list(texts)
        vecs=self.model.encode(
            payload,
            batch_size=self.batch_size,
            normalize_embeddings=True,     # so a dot product is cosine similarity
            show_progress_bar=len(payload) > 256,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)
    
@dataclass
class Hit:
    chunk: Chunk
    score: float
 
    def render(self) -> str:
        """Format for insertion into a prompt.
 
        The header matters: a bare fragment reading 'the flags parameter must be
        zero' is not usable context. Naming the source and section makes it one.
        """
        where = f"{self.chunk.path}"
        if self.chunk.section:
            where += f" -- {self.chunk.section}"
        return f"[{where}]\n{self.chunk.text}"
 
class Index:
    """One collection: chunks plus their embedding matrix."""
 
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)
 
    def __len__(self) -> int:
        return len(self.chunks)
 
    def search(self, query_vec: np.ndarray, k: int) -> list[Hit]:
        if len(self) == 0 or k <= 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        # Vectors are normalised at encode time, so the dot product is cosine.
        scores = self.vectors @ q
        k = min(k, len(self))
        # argpartition finds the top k without sorting all 3,369.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [Hit(self.chunks[i], float(scores[i])) for i in top]
 
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self.vectors,
            chunk_json=np.array([c.to_json() for c in self.chunks], dtype=object),
        )
 
    @classmethod
    def load(cls, path: Path) -> "Index":
        if not path.is_file():
            raise FileNotFoundError(
                f"No index at {path}. Run: python retrieval.py"
            )
        data = np.load(path, allow_pickle=True)
        chunks = [Chunk.from_json(s) for s in data["chunk_json"]]
        return cls(chunks, data["vectors"])
 
class Retriever:
    """Both collections, with a fixed per-collection budget."""
 
    def __init__(self, reference: Index, examples: Index,
                 embedder: Embedder | None = None):
        self.reference = reference
        self.examples = examples
        self.embedder = embedder
 
    @classmethod
    def build(cls, embedder: Embedder, chunks: list[Chunk] | None = None) -> "Retriever":
        chunks = chunks if chunks is not None else load_chunks()
        ref = [c for c in chunks if c.kind is Kind.REFERENCE]
        exs = [c for c in chunks if c.kind is Kind.EXAMPLE]
        if not ref or not exs:
            raise ValueError(
                f"Need both kinds: got {len(ref)} reference, {len(exs)} example. "
                "Rebuild the corpus with `python corpus.py`."
            )
        return cls(
            Index(ref, embedder.encode([c.text for c in ref])),
            Index(exs, embedder.encode([c.text for c in exs])),
            embedder,
        )
 
    def save(self, directory: Path | None = None) -> Path:
        directory = directory or INDEX_DIR
        self.reference.save(directory / "reference.npz")
        self.examples.save(directory / "examples.npz")
        return directory
 
    @classmethod
    def load(cls, directory: Path | None = None,
             embedder: Embedder | None = None) -> "Retriever":
        directory = directory or INDEX_DIR
        return cls(
            Index.load(directory / "reference.npz"),
            Index.load(directory / "examples.npz"),
            embedder,
        )
 
    def retrieve(self, query: str, k_reference: int | None = None,
                 k_examples: int | None = None) -> list[Hit]:
        if self.embedder is None:
            raise ValueError("Retriever has no embedder; pass one to load().")
        k_ref = RETRIEVAL.top_k_reference if k_reference is None else k_reference
        k_ex = RETRIEVAL.top_k_examples if k_examples is None else k_examples
 
        q = self.embedder.encode([build_query(query)], is_query=True)[0]
        # Reference first: it is the half the paper found reliably helpful,
        # and prompt order matters for what the model attends to.
        return self.reference.search(q, k_ref) + self.examples.search(q, k_ex)
 
    def retrieve_random(self, task_uid: str, k_reference: int | None = None,
                        k_examples: int | None = None) -> list[Hit]:
        """The control arm. Same shape as retrieve(), zero relevance.
 
        Seeded from the task id, so rerunning the experiment retrieves the same
        irrelevant chunks and the arm stays reproducible.
        """
        k_ref = RETRIEVAL.top_k_reference if k_reference is None else k_reference
        k_ex = RETRIEVAL.top_k_examples if k_examples is None else k_examples
 
        seed = int(hashlib.sha256(f"rand:{task_uid}".encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
 
        hits: list[Hit] = []
        for index, k in ((self.reference, k_ref), (self.examples, k_ex)):
            if len(index) == 0 or k <= 0:
                continue
            for i in rng.sample(range(len(index)), min(k, len(index))):
                hits.append(Hit(index.chunks[i], score=0.0))
        return hits
 
def build_query(text: str) -> str:
    """Trim a reference module down to what bge-large can actually read.
 
    Keeps the head, which carries the class name and forward body, over the tail,
    which is usually shape constants and get_inputs boilerplate.
    """
    text = text.strip()
    if len(text) <= MAX_QUERY_CHARS:
        return text
    return text[:MAX_QUERY_CHARS].rsplit("\n", 1)[0]
 
def render_hits(hits: list[Hit]) -> list[str]:
    """Format for prompts.build_prompt(retrieved=...)."""
    return [h.render() for h in hits]
 
if __name__ == "__main__":
    print("Loading corpus")
    chunks = load_chunks()
    print(f"  {len(chunks)} chunks")
 
    print(f"Loading {RETRIEVAL.embed_model} (first run downloads ~1.3 GB)")
    embedder = BGEEmbedder()
 
    print("Embedding...")
    retriever = Retriever.build(embedder, chunks)
    out = retriever.save()
 
    print(f"  reference: {len(retriever.reference)} chunks")
    print(f"  examples:  {len(retriever.examples)} chunks")
    print(f"saved to {out}")
 
    demo = "Write a HIP kernel that computes a sum reduction over a float array."
    print(f"\nSanity check -- query: {demo!r}")
    for h in retriever.retrieve(demo):
        label = "ref" if h.chunk.kind is Kind.REFERENCE else "ex "
        print(f"  {label} {h.score:.3f}  {h.chunk.path} -- {h.chunk.section[:50]}")