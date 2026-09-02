from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum

class Severity(str, Enum):
    BREAKS = "breaks_on_nvidia"    
    RISKY = "risky_on_nvidia"        
    PORTABLE = "portable"            

class Finding(str, Enum):
    HARDCODED_WAVE64 = "hardcoded_wavefront_64"
    AMD_BUILTIN = "amd_only_builtin"
    CUDA_LEAK = "cuda_api_in_hip_kernel"
    BALLOT_32BIT_MASK = "ballot_mask_assumes_32_lanes"
    USES_WARPSIZE = "uses_warpsize_builtin"

@dataclass
class LintHit:
    finding: Finding
    severity: Severity
    line_no: int
    line: str
    note: str

@dataclass
class LintReport:
    hits: list[LintHit] = field(default_factory=list)

    @property
    def breaks(self) -> bool:
        """True if a correctness failure is plausibly a platform artifact."""
        return any(h.severity is Severity.BREAKS for h in self.hits)

    @property
    def risky(self) -> bool:
        return any(h.severity is Severity.RISKY for h in self.hits)

    @property
    def portable_signal(self) -> bool:
        return any(h.finding is Finding.USES_WARPSIZE for h in self.hits)

    def categories(self) -> set[Finding]:
        return {h.finding for h in self.hits}

_WAVE_CONTEXT = re.compile(
    r"""(
        __shfl\w*\s*\([^;]*?,\s*64\s*\)      | # __shfl_down(v, off, 64)
        \b(?:lane|wave|wavefront|warp)\w*\s*[=<>!]{1,2}\s*64\b |
        \b(?:tid|threadIdx\.x|idx)\s*[%&]\s*(?:64|63)\b |
        \b\w*(?:offset|stride|delta)\w*\s*=\s*(?:64|32)\s*;[^\n]*(?:reduc|shfl|wave)
    )""",
    re.VERBOSE | re.IGNORECASE,
)
_WAVE_REDUCTION = re.compile(
    r"for\s*\([^;]*;\s*\w+\s*>\s*0\s*;\s*\w+\s*(?:>>=|/=)\s*2\s*\)", re.IGNORECASE
)
_INIT_64 = re.compile(r"=\s*(?:64\s*/\s*2|32)\b")

_AMD_BUILTIN = re.compile(r"__builtin_amdgcn_\w+|__AMDGCN_WAVEFRONT_SIZE|__gfx\d+__")

# The model leaking CUDA into what should be HIP. These compile fine on the
# NVIDIA backend (they ARE CUDA), so only this lint catches them -- on real
# AMD hardware they would be hard errors. Critical for honest reporting.
_CUDA_LEAK = re.compile(
    r"\b(cudaMalloc|cudaFree|cudaMemcpy\w*|cudaDeviceSynchronize|cudaStream\w*|"
    r"cudaGetLastError|cudaEventCreate|cudaMemset|__activemask)\b"
)

_BALLOT_32 = re.compile(r"__ballot\w*\s*\([^)]*\)[^;\n]*(?:0x[fF]{8}\b|\b4294967295\b|==\s*~0u)")

_WARPSIZE = re.compile(r"\bwarpSize\b")

_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

def _strip_comments(src: str) -> list[str]:
    """Blank out comments but keep line numbering intact."""
    cleaned = _BLOCK_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), src)
    cleaned = _LINE_COMMENT.sub("", cleaned)
    return cleaned.splitlines()

def lint_kernel(source: str) -> LintReport:
    """Scan HIP source for assumptions that differ between AMD and NVIDIA."""
    report = LintReport()
    lines = _strip_comments(source)

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        if _WAVE_CONTEXT.search(line):
            report.hits.append(LintHit(
                Finding.HARDCODED_WAVE64, Severity.BREAKS, i, stripped,
                "Literal 64 used as wavefront width. NVIDIA warps are 32 wide, "
                "so lanes 32-63 do not exist and results will be wrong.",
            ))

        if _AMD_BUILTIN.search(line):
            report.hits.append(LintHit(
                Finding.AMD_BUILTIN, Severity.BREAKS, i, stripped,
                "AMD-only compiler builtin. Has no NVIDIA equivalent; build fails.",
            ))

        if _CUDA_LEAK.search(line):
            report.hits.append(LintHit(
                Finding.CUDA_LEAK, Severity.RISKY, i, stripped,
                "CUDA API called from HIP source. Compiles on the NVIDIA backend "
                "because it IS CUDA, but would fail on real AMD hardware. Counts "
                "as the model failing to write HIP.",
            ))

        if _BALLOT_32.search(line):
            report.hits.append(LintHit(
                Finding.BALLOT_32BIT_MASK, Severity.RISKY, i, stripped,
                "Ballot result masked against a 32-bit constant. HIP __ballot "
                "returns 64 bits; the upper half is being discarded.",
            ))

        if _WARPSIZE.search(line):
            report.hits.append(LintHit(
                Finding.USES_WARPSIZE, Severity.PORTABLE, i, stripped,
                "Uses the warpSize builtin, which resolves per-platform. Portable.",
            ))

    for i, line in enumerate(lines, start=1):
        if _WAVE_REDUCTION.search(line) and _INIT_64.search(line):
            report.hits.append(LintHit(
                Finding.HARDCODED_WAVE64, Severity.BREAKS, i, line.strip(),
                "Reduction loop seeded from a 64-wide wavefront. On NVIDIA the "
                "first iterations read lanes that do not exist.",
            ))

    return report


def classify_failure(source: str, correct: bool) -> str:
    """Bucket a correctness result for per-arm reporting.

    Returns one of: 'correct', 'platform_artifact', 'model_error'.
    Only wrong answers are attributed -- a kernel that passes is simply correct,
    even if it contains risky constructs.
    """
    if correct:
        return "correct"
    report = lint_kernel(source)
    return "platform_artifact" if report.breaks else "model_error"