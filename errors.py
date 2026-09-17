from __future__ import annotations
import re
from enum import Enum

class CompileError(str, Enum):
    UNKNOWN_HIP_API = "unknown_hip_api"        
    MISSING_INCLUDE = "missing_include"
    HOST_DEVICE_MISUSE = "host_device_misuse"  
    TYPE_MISMATCH = "type_mismatch"
    LAUNCH_SYNTAX = "launch_syntax"            
    TORCH_BINDING = "torch_binding"            
    SYNTAX = "syntax"
    LINK = "link"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

class RuntimeError_(str, Enum):
    ILLEGAL_MEMORY = "illegal_memory_access"
    SHAPE_MISMATCH = "shape_mismatch"
    NUMERIC_MISMATCH = "numeric_mismatch"      
    LAUNCH_FAILURE = "launch_failure"
    OOM = "out_of_memory"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

_COMPILE_PATTERNS: list[tuple[CompileError, re.Pattern]] = [
    (CompileError.UNKNOWN_HIP_API, re.compile(
        r"identifier\s+[\"']hip\w+[\"']\s+is undefined|"
        r"[\"']hip\w+[\"']\s+was not declared|"
        r"use of undeclared identifier\s+[\"']hip\w+[\"']", re.I)),
    (CompileError.MISSING_INCLUDE, re.compile(
        r"(?:fatal error|error):\s*[\"'<][\w/\.]+[\"'>]:?\s*No such file", re.I)),
    (CompileError.HOST_DEVICE_MISUSE, re.compile(
        r"calling a __host__ function.*from a __global__|"
        r"cannot be called from a __device__|"
        r"__host__ function.*is not allowed|"
        r"a __device__ variable.*cannot be (?:directly )?(?:read|written)", re.I)),
    (CompileError.LAUNCH_SYNTAX, re.compile(
        r"hipLaunchKernelGGL.*(?:too few|too many|no matching)|"
        r"expected an expression.*<<<|kernel launch.*argument", re.I)),
    (CompileError.TORCH_BINDING, re.compile(
        r"pybind11|PYBIND11_MODULE|torch::jit|TORCH_EXTENSION_NAME|"
        r"no member named\s+[\"']\w+[\"']\s+in\s+[\"']at::Tensor", re.I)),
    (CompileError.TYPE_MISMATCH, re.compile(
        r"no (?:suitable )?conversion|cannot convert|"
        r"incompatible (?:types|operands|argument)|"
        r"no matching function for call|argument of type", re.I)),
    (CompileError.LINK, re.compile(
        r"undefined reference to|ld returned|undefined symbol|"
        r"multiple definition of", re.I)),
    (CompileError.TIMEOUT, re.compile(r"\btimed? ?out\b|TimeoutExpired", re.I)),
    (CompileError.SYNTAX, re.compile(
        r"expected\s+[\"'][^\"']+[\"']|"
        r"error: expected|parse error|"
        r"unterminated|too many errors emitted", re.I)),
]

_RUNTIME_PATTERNS: list[tuple[RuntimeError_, re.Pattern]] = [
    (RuntimeError_.OOM, re.compile(
        r"out of memory|hipErrorOutOfMemory|CUDA out of memory", re.I)),
    (RuntimeError_.ILLEGAL_MEMORY, re.compile(
        r"illegal memory access|hipErrorIllegalAddress|"
        r"misaligned address|device-side assert", re.I)),
    (RuntimeError_.LAUNCH_FAILURE, re.compile(
        r"launch fail|invalid configuration argument|"
        r"hipErrorLaunchFailure|invalid device function", re.I)),
    (RuntimeError_.SHAPE_MISMATCH, re.compile(
        r"size mismatch|shape.*mismatch|must match the size|"
        r"expected .*but got .* at dimension|The size of tensor", re.I)),
    (RuntimeError_.TIMEOUT, re.compile(r"\btimed? ?out\b|TimeoutExpired", re.I)),
]


def classify_compile(stderr: str) -> CompileError:
    """Bucket a compiler failure. Empty stderr means it built."""
    if not stderr or not stderr.strip():
        return CompileError.UNKNOWN
    for label, pattern in _COMPILE_PATTERNS:
        if pattern.search(stderr):
            return label
    return CompileError.UNKNOWN

def classify_runtime(stderr: str, *, values_differed: bool = False) -> RuntimeError_:
    """Bucket a runtime failure.

    values_differed=True means the kernel ran to completion but produced numbers
    outside tolerance. That is the most interesting bucket -- it is the model
    understanding HIP syntax but not the maths, and it is where wavefront-width
    bugs land.
    """
    for label, pattern in _RUNTIME_PATTERNS:
        if pattern.search(stderr or ""):
            return label
    if values_differed:
        return RuntimeError_.NUMERIC_MISMATCH
    return RuntimeError_.UNKNOWN

REPAIRABLE_COMPILE = {
    CompileError.UNKNOWN_HIP_API,
    CompileError.MISSING_INCLUDE,
    CompileError.HOST_DEVICE_MISUSE,
    CompileError.TYPE_MISMATCH,
    CompileError.LAUNCH_SYNTAX,
    CompileError.SYNTAX,
    CompileError.TORCH_BINDING,
}

REPAIRABLE_RUNTIME = {
    RuntimeError_.ILLEGAL_MEMORY,
    RuntimeError_.SHAPE_MISMATCH,
    RuntimeError_.NUMERIC_MISMATCH,
    RuntimeError_.LAUNCH_FAILURE,
}

def is_repairable(label: CompileError | RuntimeError_) -> bool:
    return label in REPAIRABLE_COMPILE or label in REPAIRABLE_RUNTIME

def truncate_for_prompt(stderr: str, max_lines: int = 25) -> str:
    """Compiler output is enormous and mostly template noise.
    Keep the lines that actually carry the diagnostic so the repair prompt stays
    inside the context budget.
    """
    lines = [ln for ln in (stderr or "").splitlines() if ln.strip()]
    signal = [ln for ln in lines if re.search(r"\berror\b|\bfatal\b", ln, re.I)]
    chosen = signal if signal else lines
    if len(chosen) <= max_lines:
        return "\n".join(chosen)
    head = chosen[: max_lines - 5]
    tail = chosen[-5:]
    return "\n".join(head + [f"... ({len(chosen) - max_lines} lines omitted) ..."] + tail)