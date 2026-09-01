from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

PAPER_LEVELS = (1, 2, 3)

LEVEL_NAMES = {
    1: "single operators",
    2: "fused operators",
    3: "full architectures",
    4: "hf model blocks",
}

_FILENAME = re.compile(r"^(\d+)_(.+)\.py$")

@dataclass(frozen=True)
class Task:
    level: int
    problem_id: int
    name: str
    path: Path

    @property
    def uid(self) -> str:
        """Stable identity. Used for the split hash, so it must never change."""
        return f"L{self.level}_P{self.problem_id}"

    @property
    def source(self) -> str:
        return self.path.read_text()

    def __repr__(self) -> str: 
        return f"Task({self.uid}: {self.name})"


def load_tasks(kernelbench_root: str | Path, levels=PAPER_LEVELS) -> list[Task]:
    """Load tasks from a KernelBench checkout.

    Expects <root>/KernelBench/level{N}/{problem_id}_{Name}.py
    """
    root = Path(kernelbench_root)
    task_dir = root / "KernelBench" if (root / "KernelBench").is_dir() else root

    tasks: list[Task] = []
    for level in levels:
        level_dir = task_dir / f"level{level}"
        if not level_dir.is_dir():
            raise FileNotFoundError(
                f"No level{level} directory under {task_dir}. "
                "Point kernelbench_root at a KernelBench checkout."
            )
        for f in level_dir.glob("*.py"):
            m = _FILENAME.match(f.name)
            if not m:
                continue
            tasks.append(Task(
                level=level,
                problem_id=int(m.group(1)),
                name=m.group(2),
                path=f,
            ))

    tasks.sort(key=lambda t: (t.level, t.problem_id))
    return tasks


def _split_score(uid: str, salt: str) -> float:
    """Map a task uid to a stable value in [0, 1).

    Hashing rather than shuffling means the split survives adding tasks, running on another machine, or reordering the filesystem. Python's built-in hash()
    is salted per process and would silently change between runs.
    """
    digest = hashlib.sha256(f"{salt}:{uid}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_tasks(
    tasks: list[Task],
    train_frac: float = 0.6,
    salt: str = "hipkb-v1",
) -> tuple[list[Task], list[Task]]:
    """Stratified, deterministic train/test split.
    Stratified by level because the levels differ enormously in difficulty an unstratified split could hand us a test set that is mostly level 3 and make every arm look broken.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

    train: list[Task] = []
    test: list[Task] = []

    by_level: dict[int, list[Task]] = {}
    for t in tasks:
        by_level.setdefault(t.level, []).append(t)

    for level, group in sorted(by_level.items()):
        ranked = sorted(group, key=lambda t: _split_score(t.uid, salt))
        cut = round(len(ranked) * train_frac)
        train.extend(ranked[:cut])
        test.extend(ranked[cut:])

    train.sort(key=lambda t: (t.level, t.problem_id))
    test.sort(key=lambda t: (t.level, t.problem_id))
    return train, test


def split_summary(train: list[Task], test: list[Task]) -> str:
    levels = sorted({t.level for t in train} | {t.level for t in test})
    lines = [f"{'level':<8}{'train':>7}{'test':>7}{'total':>7}  description"]
    for lv in levels:
        tr = sum(1 for t in train if t.level == lv)
        te = sum(1 for t in test if t.level == lv)
        lines.append(f"{lv:<8}{tr:>7}{te:>7}{tr + te:>7}  {LEVEL_NAMES.get(lv, '')}")
    lines.append(f"{'all':<8}{len(train):>7}{len(test):>7}{len(train) + len(test):>7}")
    return "\n".join(lines)


def assert_disjoint(train: list[Task], test: list[Task]) -> None:
    
    overlap = {t.uid for t in train} & {t.uid for t in test}
    if overlap:
        raise AssertionError(
            f"Train/test contamination: {len(overlap)} shared tasks, "
            f"e.g. {sorted(overlap)[:5]}"
        )