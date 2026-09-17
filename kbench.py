from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any
from config import BUILD_DIR, EVAL, HIP_CFLAGS, KERNELBENCH_ROOT, Profile, detect_profile
from errors import (CompileError, RuntimeError_, classify_compile, classify_runtime, truncate_for_prompt,)
from lint import LintReport, lint_kernel

def build_preamble(cflags: list[str] | None = None) -> str:
    flags = cflags if cflags is not None else HIP_CFLAGS
    return (
        "# --- injected by hipkb.kbench: do not edit ---\n"
        f"HIP_CFLAGS = {flags!r}\n"
        "# --- end injected ---\n"
    )

PREAMBLE_MARKER = "# --- injected by hipkb.kbench"

def prepare_source(model_source: str, cflags: list[str] | None = None) -> str:
    """Prepend the flag definition to generated code.
    Idempotent: calling twice will not stack two preambles, which matters
    because repair rounds feed a previously-prepared source back through.
    """
    if PREAMBLE_MARKER in model_source:
        return model_source
    return build_preamble(cflags) + "\n" + model_source


def strip_preamble(prepared_source: str) -> str:
    """Remove the preamble again.
    Needed whenever generated code is shown to the model -- in a repair prompt,
    or as an SFT training target. Training on our injected boilerplate would
    teach the model to emit it, and then it would appear twice.
    """
    lines = prepared_source.splitlines(keepends=True)
    out, inside = [], False
    for ln in lines:
        if ln.startswith(PREAMBLE_MARKER):
            inside = True
            continue
        if inside and ln.startswith("# --- end injected ---"):
            inside = False
            continue
        if not inside:
            out.append(ln)
    return "".join(out).lstrip("\n")

@dataclass
class KernelResult:
    """Outcome of evaluating one generated kernel.
    Deliberately richer than KernelBench's own result type, because the whole
    point of this project is the breakdown rather than the single number.
    """
    task_uid: str
    arm: str
    sample_idx: int = 0
    repair_round: int = 0

    compiled: bool = False
    correct: bool = False

    runtime_ms: float | None = None
    baseline_ms: float | None = None
    timing_trusted: bool = False

    compile_error: CompileError | None = None
    runtime_error: RuntimeError_ | None = None
    raw_error: str = ""

    lint: LintReport | None = None
    compliance_issues: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def speedup(self) -> float | None:
        """Baseline over ours. Above 1.0 means we are faster.
        None when timing is untrusted, so a laptop number can never silently
        reach a fast_p table.
        """
        if not (self.timing_trusted and self.runtime_ms and self.baseline_ms):
            return None
        if self.runtime_ms <= 0:
            return None
        return self.baseline_ms / self.runtime_ms

    @property
    def attribution(self) -> str:
        """Why did this fail? 'correct', 'platform_artifact', or 'model_error'.
        A wrong answer from a kernel that hardcodes a 64-wide wavefront is our
        hardware's fault, not the model's. Folding those into the correctness
        rate would understate every retrieval arm, because the RAG corpus is AMD
        documentation and AMD documentation teaches wave-64.
        """
        if self.correct:
            return "correct"
        if not self.compiled:
            return "compile_failure"
        if self.lint is not None and self.lint.breaks:
            return "platform_artifact"
        return "model_error"

    @property
    def repairable(self) -> bool:
        from errors import is_repairable
        label = self.compile_error or self.runtime_error
        return label is not None and is_repairable(label)

    def error_for_prompt(self) -> str:
        return truncate_for_prompt(self.raw_error)

def verify_upstream_assumptions(kernelbench_root=None) -> list[str]:
    """Return a list of broken assumptions. Empty means we are safe to proceed."""
    root = kernelbench_root or KERNELBENCH_ROOT
    eval_py = root / "src" / "kernelbench" / "eval.py"
    problems: list[str] = []

    if not eval_py.is_file():
        return [f"Cannot find {eval_py}. Is KERNELBENCH_ROOT correct?"]

    src = eval_py.read_text()
    # 1. The guard we are routing around still exists and still keys off the
    #    "hip" label rather than inspecting the code. If upstream starts
    #    sniffing for hip* API calls under backend="cuda", our approach dies.
    if not re.search(r'backend_lower\s*==\s*["\']hip["\'].*vendor\s*!=\s*["\']amd["\']', src, re.S):
        problems.append(
            "The HIP/AMD vendor guard in eval.py has changed shape. Re-read it "
            "before trusting backend='cuda' as the HIP-on-NVIDIA route."
        )
    # 2. cuda must not be routed down the tempfile path -- our generated code
    #    calls load_inline itself and expects the plain exec path.
    m = re.search(r"uses_tempfile\s*=\s*backend\.lower\(\)\s*in\s*(\[[^\]]*\])", src)
    if not m:
        problems.append("Could not find the uses_tempfile assignment in eval.py.")
    elif "cuda" in m.group(1):
        problems.append(
            "eval.py now routes the cuda backend through the tempfile loader. "
            "Generated kernels will not load the way this adapter expects."
        )
    # 3. The exec namespace must still be a plain dict that generated code runs
    #    inside, or the preamble will not be visible to it.
    if "exec(model_custom_src, context)" not in src:
        problems.append(
            "load_custom_model no longer execs generated code into a context "
            "dict. The HIP_CFLAGS preamble may not be in scope."
        )
    return problems

def kernelbench_importable() -> bool:
    try:
        import kernelbench  
        return True
    except ImportError:
        return False

# Evaluation
def evaluate_kernel(
    task_uid: str,
    arm: str,
    reference_source: str,
    model_source: str,
    *,
    sample_idx: int = 0,
    repair_round: int = 0,
    profile: Profile | None = None,
    device: int = 0,
    measure_time: bool | None = None,
) -> KernelResult:
    """Compile, verify, and optionally time one generated kernel.
    'measure_time' defaults to whether this machine's timings are trusted. On
    the laptop that means compile-and-verify only, which is both faster and
    honest -- there is no point timing on hardware whose numbers we will refuse
    to report.
    """
    from prompts import prompt_compliance

    profile = profile or detect_profile()
    if measure_time is None:
        measure_time = profile.trust_timing

    result = KernelResult(
        task_uid=task_uid, arm=arm,
        sample_idx=sample_idx, repair_round=repair_round,
    )

    # Static checks first. These need no GPU and cost nothing, and their output
    # is what lets us attribute a later failure correctly.
    clean_source = strip_preamble(model_source)
    result.lint = lint_kernel(clean_source)
    result.compliance_issues = prompt_compliance(clean_source)

    prepared = prepare_source(model_source)

    try:
        from kernelbench.eval import eval_kernel_against_ref
    except ImportError as e:
        result.raw_error = f"KernelBench not importable: {e}"
        result.compile_error = CompileError.UNKNOWN
        return result

    try:
        kb = eval_kernel_against_ref(
            original_model_src=reference_source,
            custom_model_src=prepared,
            # See the module docstring: on NVIDIA, HIP compiles as CUDA.
            backend="cuda",
            measure_performance=measure_time,
            num_correct_trials=EVAL.n_correctness_trials,
            num_perf_trials=EVAL.n_timing_trials if measure_time else 0,
            build_dir=str(BUILD_DIR / task_uid / arm / f"s{sample_idx}r{repair_round}"),
            device=device,
        )
    except Exception as e:  # noqa: BLE001 -- a bad kernel can raise anything
        result.raw_error = str(e)
        result.compile_error = classify_compile(str(e))
        if result.compile_error is CompileError.UNKNOWN:
            result.runtime_error = classify_runtime(str(e))
        return result

    if kb is None:
        # KernelBench returns None on lock-file contention, which is not a
        # kernel failure. Surfacing it as one would corrupt the numbers.
        result.raw_error = "KernelBench returned None (build lock contention); retry"
        result.metadata["retryable"] = True
        return result

    return _absorb(result, kb, measure_time, profile)


def _absorb(
    result: KernelResult, kb: Any, measure_time: bool, profile: Profile
) -> KernelResult:
    """Fold a KernelBench result object into ours. Split out to be testable
    without a GPU, using a stub."""
    result.compiled = bool(getattr(kb, "compiled", False))
    result.correct = bool(getattr(kb, "correctness", False))
    meta = dict(getattr(kb, "metadata", {}) or {})
    result.metadata.update(meta)

    if not result.compiled:
        err = str(meta.get("compilation_error", "")) or str(meta.get("compilation_error_name", ""))
        result.raw_error = err
        result.compile_error = classify_compile(err)
    elif not result.correct:
        err = str(meta.get("runtime_error", "")) or str(meta.get("correctness_issue", ""))
        result.raw_error = err
        # A kernel that ran to completion but produced wrong numbers reports no
        # exception at all -- that is the numeric-mismatch bucket, and it is
        # where wavefront-width bugs land.
        result.runtime_error = classify_runtime(err, values_differed=not err)

    if measure_time:
        runtime = getattr(kb, "runtime", None)
        if runtime is not None and runtime > 0:
            result.runtime_ms = float(runtime)
            result.timing_trusted = profile.trust_timing

    return result