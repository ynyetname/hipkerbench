from __future__ import annotations
import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config import EVAL, GENERATIONS_DIR, RESULTS_DIR, detect_profile, ensure_dirs
from generate import GenerationRecord, load_records
from kbench import evaluate_kernel
from tasks import Task, load_tasks

BASELINE_FILE = "baselines.json"

@dataclass
class TimedResult:
    """One generated kernel, timed. Only kernels that already passed appear."""
    task_uid: str
    level: int
    arm: str
    sample_idx: int
    repair_round: int
    correct: bool
    runtime_ms: float | None = None
    baseline_ms: float | None = None
    timing_trusted: bool = False
    attribution: str = ""
    error: str = ""

    @property
    def key(self) -> str:
        return f"{self.task_uid}|{self.arm}|{self.sample_idx}"

    @property
    def speedup(self) -> float | None:
        if not (self.timing_trusted and self.runtime_ms and self.baseline_ms):
            return None
        if self.runtime_ms <= 0:
            return None
        return self.baseline_ms / self.runtime_ms

    def to_json(self) -> str:
        d = asdict(self)
        d["speedup"] = self.speedup
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "TimedResult":
        d = json.loads(line)
        d.pop("speedup", None)
        return cls(**d)

class BaselineCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, float] = {}
        if path.is_file():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    def get(self, task: Task, device: int = 0) -> float:
        if task.uid in self.data:
            return self.data[task.uid]
        ms = time_reference(task, device=device)
        self.data[task.uid] = ms
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        return ms

def time_reference(task: Task, device: int = 0,
                   n_warmup: int | None = None,
                   n_trials: int | None = None) -> float:
    """Wall-clock time of the reference PyTorch module, in milliseconds.

    Uses CUDA events rather than time.time(): kernel launches are asynchronous,
    so a host-side timer measures how long it took to *queue* the work, not to
    run it. That mistake makes everything look 100x faster than it is.
    """
    import torch

    n_warmup = EVAL.n_warmup if n_warmup is None else n_warmup
    n_trials = EVAL.n_timing_trials if n_trials is None else n_trials

    ns: dict = {}
    exec(task.source, ns)
    dev = torch.device(f"cuda:{device}")

    model = ns["Model"](*ns["get_init_inputs"]()).to(dev)
    model.eval()
    inputs = [x.to(dev) if torch.is_tensor(x) else x for x in ns["get_inputs"]()]

    with torch.no_grad():
        for _ in range(n_warmup):
            model(*inputs)
        torch.cuda.synchronize(dev)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(n_trials):
            start.record()
            model(*inputs)
            end.record()
            torch.cuda.synchronize(dev)
            times.append(start.elapsed_time(end))

    times.sort()
    # Median, not mean: a single scheduler hiccup or a clock boost transition
    # skews a mean badly over 100 trials.
    return times[len(times) // 2]

def timeable(record: GenerationRecord) -> bool:
    """Only kernels that actually produced the right answer are worth timing."""
    return any(a.correct for a in record.attempts)

def winning_attempt(record: GenerationRecord):
    for a in record.attempts:
        if a.correct:
            return a
    return None

def evaluate_arm(
    arm: str,
    tasks_by_uid: dict[str, Task],
    *,
    gen_dir: Path | None = None,
    out_dir: Path | None = None,
    device: int = 0,
    verbose: bool = True,
) -> Path:
    """Time every passing kernel for one arm. Resumable, like generation."""
    gen_dir = gen_dir or GENERATIONS_DIR
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = detect_profile()
    if not profile.trust_timing and verbose:
        print(f"[{arm}] WARNING: {profile.name} timings are not trusted. "
              "Kernels will be re-verified but no speedups recorded.")

    records = load_records(gen_dir / f"{arm}.jsonl")
    out_path = out_dir / f"{arm}.jsonl"
    done = _completed_keys(out_path)
    baselines = BaselineCache(out_dir / BASELINE_FILE)

    todo = [r for r in records
            if f"{r.task_uid}|{arm}|{r.sample_idx}" not in done]
    if verbose:
        n_pass = sum(1 for r in todo if timeable(r))
        print(f"[{arm}] {len(todo)} records to process, {n_pass} passed and need timing")

    with out_path.open("a", encoding="utf-8") as f:
        for n, rec in enumerate(todo, start=1):
            task = tasks_by_uid.get(rec.task_uid)
            attempt = winning_attempt(rec)

            if task is None or attempt is None:
                # Record the failure so report.py sees the full denominator.
                # Dropping failures would compute fast_p over survivors only and
                # inflate every arm.
                res = TimedResult(
                    task_uid=rec.task_uid, level=rec.level, arm=arm,
                    sample_idx=rec.sample_idx,
                    repair_round=len(rec.attempts) - 1,
                    correct=False,
                    attribution=(rec.attempts[-1].attribution if rec.attempts else "unknown"),
                )
                f.write(res.to_json() + "\n")
                f.flush()
                continue

            try:
                baseline_ms = baselines.get(task, device=device)
                kr = evaluate_kernel(
                    task_uid=task.uid, arm=arm,
                    reference_source=task.source, model_source=attempt.code,
                    sample_idx=rec.sample_idx, repair_round=attempt.round,
                    measure_time=True, device=device,
                )
                res = TimedResult(
                    task_uid=task.uid, level=rec.level, arm=arm,
                    sample_idx=rec.sample_idx, repair_round=attempt.round,
                    correct=kr.correct,
                    runtime_ms=kr.runtime_ms,
                    baseline_ms=baseline_ms if kr.timing_trusted else None,
                    timing_trusted=kr.timing_trusted,
                    attribution=kr.attribution,
                )
            except Exception as e:  # noqa: BLE001
                res = TimedResult(
                    task_uid=rec.task_uid, level=rec.level, arm=arm,
                    sample_idx=rec.sample_idx,
                    repair_round=attempt.round if attempt else 0,
                    correct=False, error=f"{type(e).__name__}: {e}"[:500],
                    attribution="eval_error",
                )

            f.write(res.to_json() + "\n")
            f.flush()

            if verbose:
                sp = res.speedup
                tag = f"{sp:.2f}x" if sp else ("ok" if res.correct else "--")
                print(f"  [{n}/{len(todo)}] {res.task_uid} s{res.sample_idx} {tag}")

    return out_path

def _completed_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                keys.add(TimedResult.from_json(line).key)
            except (json.JSONDecodeError, TypeError):
                continue
    return keys

def load_results(path: Path) -> list[TimedResult]:
    if not path.is_file():
        raise FileNotFoundError(f"No results at {path}. Run evaluate.py first.")
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(TimedResult.from_json(line))
                except (json.JSONDecodeError, TypeError):
                    continue
    return out

@dataclass
class FastP:
    arm: str
    level: int | None
    threshold: float
    pass_at_1: float
    best_at_k: float
    n_tasks: int
    n_samples: int
    timing_available: bool = True
    notes: str = ""

def fast_p(
    results: list[TimedResult],
    thresholds: tuple[float, ...] | None = None,
    level: int | None = None,
) -> list[FastP]:
    """fast_p at each threshold, reported both ways.

    At p=0 this is the correctness rate, by KernelBench's own definition, and it
    needs no timing at all -- which is why a laptop run still produces a usable
    p=0 row even though every speedup row is withheld.
    """
    thresholds = thresholds or EVAL.fast_p_thresholds
    rows: list[FastP] = []
    if not results:
        return rows

    arm = results[0].arm
    scoped = [r for r in results if level is None or r.level == level]
    if not scoped:
        return rows

    by_task: dict[str, list[TimedResult]] = {}
    for r in scoped:
        by_task.setdefault(r.task_uid, []).append(r)

    have_timing = any(r.timing_trusted for r in scoped)

    for p in thresholds:
        def cleared(r: TimedResult) -> bool:
            if not r.correct:
                return False
            if p == 0.0:
                return True                 
            sp = r.speedup
            return sp is not None and sp > p

        per_sample = [cleared(r) for r in scoped]
        per_task = [any(cleared(r) for r in group) for group in by_task.values()]

        rows.append(FastP(
            arm=arm, level=level, threshold=p,
            pass_at_1=sum(per_sample) / len(per_sample),
            best_at_k=sum(per_task) / len(per_task),
            n_tasks=len(by_task), n_samples=len(scoped),
            timing_available=have_timing or p == 0.0,
            notes="" if (have_timing or p == 0.0) else "no trusted timing",
        ))

    return rows

def render(rows: list[FastP]) -> str:
    if not rows:
        return "(no results)"
    out = [f"{'p':>6}{'pass@1':>10}{'best@k':>10}{'tasks':>8}  notes"]
    for r in rows:
        if not r.timing_available:
            out.append(f"{r.threshold:>6.1f}{'--':>10}{'--':>10}{r.n_tasks:>8}  {r.notes}")
        else:
            out.append(f"{r.threshold:>6.1f}{r.pass_at_1:>9.1%}{r.best_at_k:>10.1%}"
                       f"{r.n_tasks:>8}")
    return "\n".join(out)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default=None,
                    help="comma-separated; default is every arm found on disk")
    ap.add_argument("--levels", default="1")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    ensure_dirs()
    profile = detect_profile()
    print(f"profile: {profile.name}, timing "
          f"{'trusted' if profile.trust_timing else 'NOT trusted'}")

    levels = tuple(int(x) for x in args.levels.split(","))
    tasks_by_uid = {t.uid: t for t in load_tasks(levels=levels)}

    if args.arms:
        arms = [a.strip() for a in args.arms.split(",")]
    else:
        arms = sorted(p.stem for p in GENERATIONS_DIR.glob("*.jsonl"))
        if not arms:
            raise SystemExit(f"No generations in {GENERATIONS_DIR}. Run generate.py first.")

    for arm in arms:
        print(f"\n=== {arm} ===")
        t0 = time.time()
        path = evaluate_arm(arm, tasks_by_uid, device=args.device)
        results = load_results(path)
        print(f"  {len(results)} results in {(time.time() - t0) / 60:.0f}m")
        for lv in levels:
            rows = fast_p(results, level=lv)
            if rows:
                print(f"\n  level {lv}:")
                print("  " + render(rows).replace("\n", "\n  "))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())