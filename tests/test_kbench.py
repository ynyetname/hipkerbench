import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PROFILES  # noqa: E402
from errors import CompileError, RuntimeError_  # noqa: E402
from kbench import (  # noqa: E402
    KernelResult, _absorb, build_preamble, prepare_source, strip_preamble,
    verify_upstream_assumptions,
)
from lint import lint_kernel  # noqa: E402

KB_ROOT = Path(os.environ.get("HIPKB_KERNELBENCH", Path.home() / "KernelBench"))
needs_kb = pytest.mark.skipif(not KB_ROOT.is_dir(), reason="KernelBench not checked out")

KERNEL = '''import torch
from torch.utils.cpp_extension import load_inline
mod = load_inline(name="m", cuda_sources="...", extra_cuda_cflags=HIP_CFLAGS)
class ModelNew(torch.nn.Module):
    pass
'''

def test_preamble_defines_the_name_the_model_references():
    prepared = prepare_source(KERNEL)
    ns: dict[str, Any] = {}
    exec(prepared.split("import torch")[0], ns)  # just the preamble
    assert "HIP_CFLAGS" in ns
    assert "-D__HIP_PLATFORM_NVIDIA__" in ns["HIP_CFLAGS"]

def test_prepare_is_idempotent():
    """Repair rounds feed prepared source back through. Two preambles would
    shadow each other and make the failure baffling to debug."""
    once = prepare_source(KERNEL)
    twice = prepare_source(once)
    assert once == twice
    assert once.count("HIP_CFLAGS = [") == 1

def test_strip_removes_exactly_the_preamble():
    assert strip_preamble(prepare_source(KERNEL)) == KERNEL

def test_strip_is_a_noop_on_unprepared_source():
    assert strip_preamble(KERNEL) == KERNEL

def test_strip_before_showing_code_to_the_model():
    """Training on injected boilerplate would teach the model to emit it, and
    then prepare_source would find a marker it did not write."""
    assert "HIP_CFLAGS = [" not in strip_preamble(prepare_source(KERNEL))
    assert "extra_cuda_cflags=HIP_CFLAGS" in strip_preamble(prepare_source(KERNEL))

def test_preamble_carries_no_machine_specific_path_into_the_kernel_body():
    """The kernel body must stay portable: compiled on the laptop, timed on
    Kaggle, where the include path differs."""
    body = strip_preamble(prepare_source(KERNEL))
    assert "-I" not in body

def test_custom_flags_are_honoured():
    p = prepare_source("class ModelNew: pass", cflags=["-DFOO", "-I/x"])
    assert "'-DFOO'" in p and "'-I/x'" in p

def test_build_preamble_is_valid_python():
    ns: dict[str, Any] = {}
    exec(build_preamble(["-a", "-b"]), ns)
    assert ns["HIP_CFLAGS"] == ["-a", "-b"]

@dataclass
class StubKB:
    """Stands in for KernelBench's result object so the absorption logic is
    testable without a GPU."""
    compiled: bool = True
    correctness: bool = True
    runtime: float | None = 2.0
    metadata: dict = field(default_factory=dict)

def _absorbed(kb, *, measure_time=True, profile="kaggle_t4"):
    return _absorb(KernelResult("L1_P1", "rag"), kb, measure_time, PROFILES[profile])

def test_laptop_timing_never_produces_a_speedup():
    """The core guard on the compile-here/time-there split. A laptop number must
    not be able to reach a fast_p table by any route."""
    r = _absorbed(StubKB(runtime=2.0), profile="laptop")
    r.baseline_ms = 4.0
    assert r.timing_trusted is False
    assert r.speedup is None

def test_kaggle_timing_produces_a_speedup():
    r = _absorbed(StubKB(runtime=2.0), profile="kaggle_t4")
    r.baseline_ms = 4.0
    assert r.speedup == pytest.approx(2.0)

def test_speedup_needs_a_baseline():
    r = _absorbed(StubKB(runtime=2.0))
    assert r.speedup is None

def test_zero_runtime_does_not_divide_by_zero():
    r = _absorbed(StubKB(runtime=0.0))
    r.baseline_ms = 4.0
    assert r.speedup is None

def test_compile_failure_is_classified():
    kb = StubKB(compiled=False, correctness=False, runtime=None, metadata={
        "compilation_error": 'error: identifier "hipMallocAsync2" is undefined'})
    r = _absorbed(kb)
    assert r.compile_error is CompileError.UNKNOWN_HIP_API
    assert r.attribution == "compile_failure"
    assert r.repairable

def test_ran_but_wrong_numbers_is_numeric_mismatch():
    """No exception, just wrong values. This is the bucket wavefront bugs
    land in, so it must not be swallowed as 'unknown'."""
    r = _absorbed(StubKB(compiled=True, correctness=False, metadata={}))
    assert r.runtime_error is RuntimeError_.NUMERIC_MISMATCH

def test_wavefront_bug_is_attributed_to_the_platform_not_the_model():
    wave64 = """
#include <hip/hip_runtime.h>
__global__ void k(float* v) {
    for (int offset = 64 / 2; offset > 0; offset >>= 1) {
        v[0] += __shfl_down(v[0], offset, 64);
    }
}
class ModelNew: pass
"""
    r = _absorbed(StubKB(compiled=True, correctness=False))
    r.lint = lint_kernel(wave64)
    assert r.attribution == "platform_artifact"

def test_plain_wrong_kernel_is_attributed_to_the_model():
    plain = "#include <hip/hip_runtime.h>\n__global__ void k(float* v){ v[0]=1.0f; }"
    r = _absorbed(StubKB(compiled=True, correctness=False))
    r.lint = lint_kernel(plain)
    assert r.attribution == "model_error"

def test_correct_kernel_is_never_attributed_as_a_failure():
    r = _absorbed(StubKB(compiled=True, correctness=True))
    r.lint = lint_kernel("__shfl_down(v, off, 64);")
    assert r.attribution == "correct", "a passing kernel is correct, risky or not"

def test_oom_is_not_offered_for_repair():
    kb = StubKB(compiled=True, correctness=False,
                metadata={"runtime_error": "HIP out of memory. Tried to allocate 20 GiB"})
    r = _absorbed(kb)
    assert r.runtime_error is RuntimeError_.OOM
    assert not r.repairable, "retrying an OOM burns generation budget for nothing"

def test_error_for_prompt_is_truncated():
    noise = "\n".join(f"note: instantiation {i}" for i in range(300))
    kb = StubKB(compiled=False, correctness=False, runtime=None,
                metadata={"compilation_error": noise + "\nerror: expected a ';'"})
    r = _absorbed(kb)
    out = r.error_for_prompt()
    assert "error: expected a ';'" in out and len(out.splitlines()) <= 25

def test_timing_skipped_when_not_measured():
    r = _absorbed(StubKB(runtime=5.0), measure_time=False)
    assert r.runtime_ms is None

@needs_kb
def test_upstream_assumptions_hold_today():
    """If this fails after a KernelBench update, re-read eval.py before
    trusting backend='cuda' as the HIP-on-NVIDIA route."""
    problems = verify_upstream_assumptions(KB_ROOT)
    assert problems == [], problems

def test_missing_kernelbench_is_reported_clearly(tmp_path):
    problems = verify_upstream_assumptions(tmp_path)
    assert len(problems) == 1 and "Cannot find" in problems[0]

def test_detects_cuda_being_routed_through_tempfile(tmp_path):
    """A synthetic future version of eval.py that would silently break us."""
    d = tmp_path / "src" / "kernelbench"
    d.mkdir(parents=True)
    (d / "eval.py").write_text(
        'if backend_lower == "hip" and vendor != "amd":\n'
        '    raise ValueError("nope")\n'
        'uses_tempfile = backend.lower() in ["triton", "cuda", "cute"]\n'
        "exec(model_custom_src, context)\n"
    )
    problems = verify_upstream_assumptions(tmp_path)
    assert any("tempfile loader" in p for p in problems)

def test_detects_removal_of_the_exec_context(tmp_path):
    d = tmp_path / "src" / "kernelbench"
    d.mkdir(parents=True)
    (d / "eval.py").write_text(
        'if backend_lower == "hip" and vendor != "amd":\n'
        '    raise ValueError("nope")\n'
        'uses_tempfile = backend.lower() in ["triton"]\n'
        "importlib.import_module(tmpmod)\n"
    )
    problems = verify_upstream_assumptions(tmp_path)
    assert any("HIP_CFLAGS preamble" in p for p in problems)