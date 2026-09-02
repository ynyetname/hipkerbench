from __future__ import annotations
import re
from dataclasses import dataclass, field

HIP_NVIDIA_EXEMPLAR = '''import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void elementwise_add_kernel(const float* a, const float* b,
                                       float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] + b[idx];
}

torch::Tensor elementwise_add_hip(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int size = a.numel();
    const int block = 256;
    const int grid = (size + block - 1) / block;
    hipLaunchKernelGGL(elementwise_add_kernel, dim3(grid), dim3(block), 0, 0,
                       a.data_ptr<float>(), b.data_ptr<float>(),
                       out.data_ptr<float>(), size);
    return out;
}
"""

cpp_decl = "torch::Tensor elementwise_add_hip(torch::Tensor a, torch::Tensor b);"

elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=cpp_decl,
    cuda_sources=hip_source,
    functions=["elementwise_add_hip"],
    extra_cuda_cflags=HIP_CFLAGS,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.op = elementwise_add

    def forward(self, a, b):
        return self.op.elementwise_add_hip(a, b)
'''

SYSTEM = (
    "You write AMD HIP GPU kernels. You are given a PyTorch module named Model. "
    "Replace its operations with custom HIP kernels and return a module named "
    "ModelNew with identical behaviour."
)

RULES = """Requirements:
- Return one complete Python file, in a single ```python code block.
- The module must be named ModelNew and keep the same __init__ and forward signature.
- Write HIP: include <hip/hip_runtime.h> and use hip* API calls, not cuda* calls.
- Put kernel code in `cuda_sources` and the function declaration in `cpp_sources`.
- Pass `extra_cuda_cflags=HIP_CFLAGS` to load_inline. HIP_CFLAGS is already
  defined in the execution environment; do not redefine it.
- Do not set os.environ["CXX"].
- No explanation outside the code block."""

PORTABILITY_HINT = """- Do not hardcode the wavefront width. Use the `warpSize` builtin, which resolves correctly per platform."""

@dataclass
class PromptSpec:
    """One experimental arm's prompt configuration."""
    name: str
    use_retrieval: bool = False
    random_retrieval: bool = False  
    one_shot: bool = True
    portability_hint: bool = False   

    def __post_init__(self):
        if self.random_retrieval and not self.use_retrieval:
            raise ValueError(
                f"arm '{self.name}': random_retrieval requires use_retrieval. "
                "The control must match the treatment's token budget."
            )

ARMS: dict[str, PromptSpec] = {
    "base":     PromptSpec("base"),
    "rag":      PromptSpec("rag", use_retrieval=True),
    "rand":     PromptSpec("rand", use_retrieval=True, random_retrieval=True),
    "sft":      PromptSpec("sft"),
    "rag_sft":  PromptSpec("rag_sft", use_retrieval=True),
    "hint":     PromptSpec("hint", use_retrieval=True, portability_hint=True),
}

def build_prompt(
    reference_source: str,
    spec: PromptSpec,
    retrieved: list[str] | None = None,
) -> list[dict[str, str]]:
    """Assemble chat messages for one task under one arm."""
    if spec.use_retrieval and not retrieved:
        raise ValueError(
            f"arm '{spec.name}' expects retrieved chunks but got none. "
            "Silently falling back to no-retrieval would corrupt the ablation."
        )

    parts: list[str] = []

    if retrieved:
        label = ("Reference material (HIP documentation and examples):"
                 if not spec.random_retrieval
                 else "Reference material:")
        joined = "\n\n---\n\n".join(retrieved)
        parts.append(f"{label}\n\n{joined}")

    if spec.one_shot:
        parts.append(
            "Here is a worked example of the expected output format:\n\n"
            f"```python\n{HIP_NVIDIA_EXEMPLAR}```"
        )

    rules = RULES + ("\n" + PORTABILITY_HINT if spec.portability_hint else "")
    parts.append(f"Optimise this PyTorch module:\n\n```python\n{reference_source}```\n\n{rules}")

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]

def build_repair_prompt(
    messages: list[dict[str, str]],
    failed_source: str,
    error_text: str,
) -> list[dict[str, str]]:
    """Append one repair turn carrying the compiler's diagnostic.

    Keeps the original context so the model still sees the task and any
    retrieved docs. `error_text` should already be truncated by
    errors.truncate_for_prompt -- raw nvcc output is mostly template noise and
    will blow the context budget.
    """
    return messages + [
        {"role": "assistant", "content": f"```python\n{failed_source}\n```"},
        {"role": "user", "content":
            f"That failed. The toolchain reported:\n\n```\n{error_text}\n```\n\n"
            "Fix the problem and return the complete corrected file in a single "
            "```python code block. No explanation."},
    ]

_FENCED = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

@dataclass
class ParseResult:
    code: str | None
    reason: str = ""
    fences_found: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.code is not None

def extract_model_new(response: str) -> ParseResult:
    """Pull the ModelNew source out of a model response.
    Failing to parse is scored as a compilation failure, so this needs to be
    generous about formatting but strict about correctness -- returning the
    wrong block would silently evaluate a snippet instead of the real answer.
    """
    if not response or not response.strip():
        return ParseResult(None, "empty response")

    blocks = [b.strip() for b in _FENCED.findall(response)]
    warnings: list[str] = []

    if blocks:
        # Prefer the last block defining ModelNew: models often show a broken
        # draft first and the corrected version last.
        with_model = [b for b in blocks if re.search(r"class\s+ModelNew\b", b)]
        if with_model:
            chosen = with_model[-1]
            if len(with_model) > 1:
                warnings.append(f"{len(with_model)} blocks defined ModelNew; took the last")
            return ParseResult(chosen, fences_found=len(blocks), warnings=warnings)
        return ParseResult(
            None, "code block present but no `class ModelNew`", len(blocks), warnings
        )

    # Unfenced fallback: some models emit bare code.
    if re.search(r"class\s+ModelNew\b", response):
        warnings.append("no code fence; used raw response")
        return ParseResult(response.strip(), fences_found=0, warnings=warnings)

    return ParseResult(None, "no code block and no `class ModelNew`")

_CUDA_API = re.compile(r"\bcuda[A-Z]\w*\s*\(")
_SETS_CXX = re.compile(r"os\.environ\s*\[\s*[\"']CXX[\"']\s*\]")

def prompt_compliance(code: str) -> list[str]:
    """Check the model followed the structural instructions.

    Not correctness -- these are things that make a kernel fail for reasons
    unrelated to whether the model understands HIP, and they should be reported
    separately rather than folded into the correctness rate.
    """
    issues = []
    if _SETS_CXX.search(code):
        issues.append("sets CXX (upstream AMD idiom; breaks the nvcc path)")
    if "cuda_sources" not in code and "load_inline" in code:
        issues.append("kernel not in cuda_sources (g++ cannot parse <<<>>>)")
    if _CUDA_API.search(code):
        issues.append("calls cuda* API instead of hip*")
    if "hip/hip_runtime.h" not in code:
        issues.append("missing #include <hip/hip_runtime.h>")
    return issues