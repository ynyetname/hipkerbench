from __future__ import annotations
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from config import CORPUS_DIR, RETRIEVAL

class Kind(str, Enum):
    REFERENCE = "reference"
    EXAMPLE = "example" 

SOURCES = [
    ("hip-docs", "https://github.com/ROCm/hip.git", "rocm-7.0.2",
     ["docs"], Kind.REFERENCE),
    ("rocm-examples", "https://github.com/ROCm/rocm-examples.git", None,
     ["HIP-Basic", "Applications", "Programming-Guide", "Tutorials"], Kind.EXAMPLE),
]

REFERENCE_EXT = {".rst", ".md", ".txt"}
EXAMPLE_EXT = {".hip", ".cpp", ".cu", ".h", ".hpp"}

SKIP_PARTS = {"sphinx", "doxygen", ".github", ".gitlab", "cmake", "_static", "_templates"}
SKIP_NAMES = {"conf.py", "LICENSE", "CMakeLists.txt", "Makefile"}

@dataclass
class Document:
    doc_id: str
    source: str          # which repo it came from
    path: str            # path within that repo
    kind: Kind
    text: str
 
    @property
    def title(self) -> str:
        return Path(self.path).stem.replace("_", " ")
    
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    kind: Kind
    text: str
    source: str
    path: str
    section: str = ""
    metadata: dict=field(default_factory=dict)
    
    def to_json(self) -> str:
        d=asdict(self)
        d["kind"]=self.kind.value
        return json.dumps(d, ensure_ascii=False)    
    @classmethod
    def from_json(cls, line:str) -> "Chunk":
        d=json.loads(line)
        d["kind"]=Kind(d["kind"])
        return cls(**d)
    
def fetch_sources(dest: Path | None=None, quiet: bool=True) -> dict[str, Path]:
    dest = dest or (CORPUS_DIR / "src")
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
 
    for name, url, branch, sparse, _kind in SOURCES:
        repo = dest / name
        out[name] = repo
        if repo.exists():
            continue
        cmd = ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse"]
        if branch:
            cmd += ["--branch", branch]
        if quiet:
            cmd.append("-q")
        cmd += [url, str(repo)]
        subprocess.run(cmd, check=True)
        subprocess.run(["git", "sparse-checkout", "set", *sparse], cwd=repo, check=True)
    return out

def _should_skip(rel: Path) -> bool:
    if rel.name in SKIP_NAMES:
        return True
    return any(part in SKIP_PARTS for part in rel.parts)

def load_documents(repos: dict[str, Path] | None = None) -> list[Document]:
    """Read every documentation and example file into Document objects"""
    repos = repos or fetch_sources()
    docs: list[Document] = []
 
    for name, _url, _branch, sparse, kind in SOURCES:
        root = repos.get(name)
        if root is None or not root.is_dir():
            continue
        wanted = REFERENCE_EXT if kind is Kind.REFERENCE else EXAMPLE_EXT
 
        for sub in sparse:
            base = root / sub
            if not base.is_dir():
                continue
            for f in sorted(base.rglob("*")):
                if not f.is_file() or f.suffix not in wanted:
                    continue
                rel = f.relative_to(root)
                if _should_skip(rel):
                    continue
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if len(text.strip()) < 100:      
                    continue
                docs.append(Document(
                    doc_id=f"{name}:{rel.as_posix()}",
                    source=name,
                    path=rel.as_posix(),
                    kind=kind,
                    text=text,
                ))
 
    return docs

# Header Splitting

_RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+_])\1{2,}\s*$")
_MD_HEADER = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
 
def split_by_headers(text: str, is_markdown: bool = False) -> list[tuple[str, str]]:
    """Split prose into (section_trail, body) pairs.
 
    The section trail is what makes a chunk interpretable on its own. A chunk
    reading "the flags parameter must be zero" is useless without knowing it
    came from "Memory management > hipMallocManaged".
    """
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    trail: list[str] = []
    levels: list[str] = []          
    buf: list[str] = []
 
    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(trail), body))
        buf.clear()
 
    i = 0
    while i < len(lines):
        line = lines[i]
 
        if is_markdown:
            m = _MD_HEADER.match(line)
            if m:
                flush()
                depth = len(m.group(1))
                trail[:] = trail[: depth - 1] + [m.group(2).strip()]
                i += 1
                continue
        else:
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            um = _RST_UNDERLINE.match(nxt)
            # An underline must be at least as long as its title to count.
            if um and line.strip() and len(nxt.strip()) >= len(line.strip()):
                flush()
                char = um.group(1)
                if char not in levels:
                    levels.append(char)
                depth = levels.index(char) + 1
                trail[:] = trail[: depth - 1] + [line.strip()]
                i += 2
                continue
 
        buf.append(line)
        i += 1
 
    flush()
    return sections
 
 
_FUNC_START = re.compile(
    r"^\s*(?:__global__|__device__|__host__|template\s*<|inline\s|static\s|"
    r"[A-Za-z_][\w:<>,\s\*&]*\s+[A-Za-z_]\w*\s*\([^;]*$)"
)
 
def split_code(text: str) -> list[tuple[str, str]]:
    """Split source into (function_name, body) pairs at brace depth zero."""
    lines = text.splitlines()
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    depth = 0
    name = "preamble"
    pending = None
 
    for line in lines:
        if depth == 0 and _FUNC_START.match(line) and buf and "".join(buf).strip():
            out.append((name, "\n".join(buf).strip()))
            buf = []
            name = pending or "function"
        if depth == 0:
            m = re.search(r"\b([A-Za-z_]\w*)\s*\(", line)
            if m:
                pending = m.group(1)
                if not buf or not "".join(buf).strip():
                    name = pending
        buf.append(line)
        depth += line.count("{") - line.count("}")
        depth = max(depth, 0)
 
    if "".join(buf).strip():
        out.append((name, "\n".join(buf).strip()))
    return [(n, b) for n, b in out if b.strip()]
 
_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", " ", ""]
 
def recursive_split(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
    separators: list[str] = None,
) -> list[str]:
    chunk_size = chunk_size or RETRIEVAL.chunk_size
    overlap = overlap if overlap is not None else RETRIEVAL.chunk_overlap
    separators = separators or _SEPARATORS
 
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
 
    sep = ""
    rest = separators
    for i, s in enumerate(separators):
        if s == "" or s in text:
            sep = s
            rest = separators[i + 1:]
            break
 
    pieces = list(text) if sep == "" else text.split(sep)
 
    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        candidate = piece if not cur else cur + sep + piece
        if len(candidate) <= chunk_size:
            cur = candidate
            continue
        if cur:
            chunks.append(cur)
            cur = (cur[-overlap:] + sep + piece) if overlap else piece
        else:
            cur = piece
        if len(cur) > chunk_size and rest:
            chunks.extend(recursive_split(cur, chunk_size, overlap, rest))
            cur = ""
 
    if cur.strip():
        chunks.append(cur)
    return [c for c in chunks if c.strip()]
 
# Pipeline

def chunk_document(doc: Document, chunk_size: int = None, overlap: int = None) -> list[Chunk]:
    chunk_size = chunk_size or RETRIEVAL.chunk_size
    overlap = overlap if overlap is not None else RETRIEVAL.chunk_overlap
 
    if doc.kind is Kind.EXAMPLE:
        sections = split_code(doc.text)
    else:
        sections = split_by_headers(doc.text, is_markdown=doc.path.endswith(".md"))
 
    chunks: list[Chunk] = []
    for section, body in sections:
        for j, piece in enumerate(recursive_split(body, chunk_size, overlap)):
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}#{len(chunks)}",
                doc_id=doc.doc_id,
                kind=doc.kind,
                text=piece,
                source=doc.source,
                path=doc.path,
                section=section,
                metadata={"part": j},
            ))
    return chunks
 
def build_corpus(chunk_size: int = None, overlap: int = None) -> list[Chunk]:
    docs = load_documents()
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(chunk_document(d, chunk_size, overlap))
    return chunks
 
def save_chunks(chunks: list[Chunk], path: Path | None = None) -> Path:
    path = path or (CORPUS_DIR / "chunks.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.to_json() + "\n")
    return path
 
def load_chunks(path: Path | None = None) -> list[Chunk]:
    path = path or (CORPUS_DIR / "chunks.jsonl")
    if not path.is_file():
        raise FileNotFoundError(
            f"No corpus at {path}. Run: python corpus.py"
        )
    with path.open(encoding="utf-8") as f:
        return [Chunk.from_json(line) for line in f if line.strip()]
 
def summarize(chunks: list[Chunk]) -> str:
    by_kind: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_kind.setdefault(c.kind.value, []).append(c)
 
    lines = [f"{'kind':<12}{'docs':>7}{'chunks':>9}{'avg chars':>11}"]
    for kind, group in sorted(by_kind.items()):
        n_docs = len({c.doc_id for c in group})
        avg = sum(len(c.text) for c in group) // max(len(group), 1)
        lines.append(f"{kind:<12}{n_docs:>7}{len(group):>9}{avg:>11}")
    total_docs = len({c.doc_id for c in chunks})
    lines.append(f"{'total':<12}{total_docs:>7}{len(chunks):>9}")
    return "\n".join(lines)
 
if __name__ == "__main__":
    print("Fetching sources")
    fetch_sources(quiet=False)
    print("Building corpus")
    chunks = build_corpus()
    path = save_chunks(chunks)
    print()
    print(summarize(chunks))
    print(f"\nsaved to {path}")
 

        