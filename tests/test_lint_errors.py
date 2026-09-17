import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from errors import (CompileError, RuntimeError_, classify_compile, classify_runtime, is_repairable, truncate_for_prompt,)
from lint import Finding, classify_failure, lint_kernel 

WAVE64_REDUCTION = """
#include <hip/hip_runtime.h>
__global__ void reduce_sum(const float* in, float* out, int n) {
    float v = (threadIdx.x < n) ? in[threadIdx.x] : 0.0f;
    // classic AMD-doc idiom: fold across a 64-wide wavefront
    for (int offset = 64 / 2; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, 64);
    }
    if (threadIdx.x % 64 == 0) out[blockIdx.x] = v;
}
"""

PORTABLE_REDUCTION = """
#include <hip/hip_runtime.h>
__global__ void reduce_sum(const float* in, float* out, int n) {
    float v = (threadIdx.x < n) ? in[threadIdx.x] : 0.0f;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, warpSize);
    }
    if (threadIdx.x % warpSize == 0) out[blockIdx.x] = v;
}
"""

CUDA_LEAK = """
#include <hip/hip_runtime.h>
void launch(float* h) {
    float* d;
    cudaMalloc(&d, 1024);           // model wrote CUDA, not HIP
    cudaMemcpy(d, h, 1024, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}
"""

AMD_ONLY = """
#include <hip/hip_runtime.h>
__global__ void k(float* o) {
    o[0] = __builtin_amdgcn_workitem_id_x();
}
"""

BENIGN_64 = """
#include <hip/hip_runtime.h>
__global__ void add(const float* a, const float* b, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] + b[i];
}
// launched with 64 threads per block -- a portable, legitimate choice
"""

COMMENTED_OUT = """
#include <hip/hip_runtime.h>
// old version used __shfl_down(v, offset, 64) before we fixed it
__global__ void k(float* o) { o[0] = 1.0f; }
"""
def test_wave64_reduction_flagged_as_breaking():
    r = lint_kernel(WAVE64_REDUCTION)
    assert r.breaks, "wavefront-64 reduction must be flagged"
    assert Finding.HARDCODED_WAVE64 in r.categories()

def test_portable_reduction_not_flagged():
    r = lint_kernel(PORTABLE_REDUCTION)
    assert not r.breaks, [h.line for h in r.hits if h.severity == "breaks_on_nvidia"]
    assert r.portable_signal, "warpSize usage should be recorded as portable"

def test_cuda_leak_is_risky_not_breaking():
    """cudaMalloc compiles and runs fine here -- it IS CUDA. It would only fail
    on real AMD hardware, so it must not be counted as a platform artifact."""
    r = lint_kernel(CUDA_LEAK)
    assert Finding.CUDA_LEAK in r.categories()
    assert r.risky and not r.breaks

def test_amd_builtin_flagged():
    assert Finding.AMD_BUILTIN in lint_kernel(AMD_ONLY).categories()

def test_benign_64_not_flagged():
    """A 64-thread block is normal and portable. Matching a bare 64 anywhere
    would flag nearly every kernel and make the signal useless."""
    r = lint_kernel(BENIGN_64)
    assert not r.breaks, f"false positive on a plain elementwise kernel: {r.hits}"

def test_comments_are_ignored():
    """Regression test. The first comment-stripper used re.DOTALL, so `//.*$`
    matched to the end of the FILE, blanking every kernel with a comment in it.
    That was a silent zero across the whole analysis."""
    r = lint_kernel(COMMENTED_OUT)
    assert not r.breaks, "lint must not fire on commented-out code"

def test_attribution_only_applies_to_failures():
    assert classify_failure(WAVE64_REDUCTION, correct=True) == "correct"
    assert classify_failure(WAVE64_REDUCTION, correct=False) == "platform_artifact"
    assert classify_failure(BENIGN_64, correct=False) == "model_error"

def test_invented_hip_api():
    err = 'kernel.hip(31): error: identifier "hipMallocManagedAsync" is undefined'
    assert classify_compile(err) is CompileError.UNKNOWN_HIP_API

def test_clang_style_undeclared_hip_api():
    err = "kernel.hip:12:5: error: use of undeclared identifier 'hipStreamCreateExt'"
    assert classify_compile(err) is CompileError.UNKNOWN_HIP_API

def test_host_pointer_in_device_code():
    err = ("kernel.hip(44): error: calling a __host__ function "
           '"at::Tensor::data_ptr<float>" from a __global__ function is not allowed')
    assert classify_compile(err) is CompileError.HOST_DEVICE_MISUSE

def test_missing_header():
    err = 'kernel.hip:1:10: fatal error: "hip/hip_runtime.h": No such file or directory'
    assert classify_compile(err) is CompileError.MISSING_INCLUDE

def test_type_mismatch():
    err = ("kernel.hip(18): error: no suitable conversion function from "
           '"at::Tensor" to "float *" exists')
    assert classify_compile(err) is CompileError.TYPE_MISMATCH

def test_link_error():
    err = "/usr/bin/ld: hip_smoke.o: undefined reference to `hipLaunchKernel'"
    assert classify_compile(err) is CompileError.LINK

def test_empty_stderr_is_unknown_not_a_crash():
    assert classify_compile("") is CompileError.UNKNOWN

def test_specific_beats_general_ordering():
    """A message with both an undefined hip API and generic syntax noise must
    bucket as the HIP API problem -- that is the one RAG can fix."""
    err = ('error: identifier "hipMemcpyDtoDAsync" is undefined\n'
           "error: expected a ';'")
    assert classify_compile(err) is CompileError.UNKNOWN_HIP_API

def test_runtime_illegal_access():
    assert classify_runtime("RuntimeError: HIP error: an illegal memory access "
                            "was encountered") is RuntimeError_.ILLEGAL_MEMORY

def test_runtime_shape_mismatch():
    assert classify_runtime("The size of tensor a (128) must match the size of "
                            "tensor b (256) at dimension 1") is RuntimeError_.SHAPE_MISMATCH

def test_clean_run_with_wrong_values():
    """No exception, just wrong numbers. This is where wavefront bugs land, so
    it must not be swallowed as 'unknown'."""
    assert classify_runtime("", values_differed=True) is RuntimeError_.NUMERIC_MISMATCH

def test_oom_is_not_repairable():
    label = classify_runtime("HIP out of memory. Tried to allocate 20.00 GiB")
    assert label is RuntimeError_.OOM
    assert not is_repairable(label), "retrying an OOM burns budget for nothing"

def test_hip_api_error_is_repairable():
    assert is_repairable(CompileError.UNKNOWN_HIP_API)

def test_truncate_keeps_error_lines_only():
    """Real nvcc output is mostly deprecation noise -- you saw hundreds of lines
    around two real errors in the smoke test."""
    noise = "\n".join([f"note: template instantiation {i}" for i in range(200)])
    err = noise + "\nkernel.hip(9): error: expected a ';'\n" + noise
    out = truncate_for_prompt(err)
    assert "error: expected a ';'" in out
    assert "template instantiation" not in out
    assert len(out.splitlines()) <= 25

def test_truncate_handles_short_input():
    assert truncate_for_prompt("error: boom") == "error: boom"
    assert truncate_for_prompt("") == ""