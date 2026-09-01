from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent

def _path_from_env(var: str, default: Path) -> Path:
    return Path(os.environ.get(var, str(default))).expanduser()

def _find_kernelbench() -> Path:

    if "HIPKB_KERNELBENCH" in os.environ:
        return Path(os.environ["HIPKB_KERNELBENCH"]).expanduser()

    here = Path(__file__).resolve().parent
    for candidate in (
        here / "KernelBench",           
        here.parent / "KernelBench",      
        REPO_ROOT / "KernelBench",
        REPO_ROOT.parent / "KernelBench",
        Path.home() / "KernelBench",
    ):
        if (candidate / "KernelBench" / "level1").is_dir():
            return candidate
    return REPO_ROOT.parent / "KernelBench"   

KERNELBENCH_ROOT = _find_kernelbench()

HIP_INCLUDE = _path_from_env("HIPKB_HIP_INCLUDE", Path.home() / "hip-nvidia" / "include")

DATA_DIR = _path_from_env("HIPKB_DATA", REPO_ROOT / "data")
CORPUS_DIR = DATA_DIR / "corpus"         
INDEX_DIR = DATA_DIR / "index"          
GENERATIONS_DIR = DATA_DIR / "generations"  
RESULTS_DIR = DATA_DIR / "results"        
SFT_DIR = DATA_DIR / "sft"
BUILD_DIR = DATA_DIR / "build"            

def ensure_dirs() -> None:
    for d in (CORPUS_DIR, INDEX_DIR, GENERATIONS_DIR, RESULTS_DIR, SFT_DIR, BUILD_DIR):
        d.mkdir(parents=True, exist_ok=True)
        
@dataclass(frozen=True)
class Profile:
    name: str
    arch: str
    trust_timing: bool
    vram_gb: float
    notes: str=""
    
    @property
    def can_train_7b(self) -> bool:
        """QLoRA on a 7B model needs roughly 10-12GB once activations are counted."""
        return self.vram_gb >= 14
    
PROFILES: dict[str, Profile] = {
    "laptop": Profile(
        name="laptop",
        arch="8.6",                
        trust_timing=False,        
        vram_gb=6.0,
        notes="compile + correctness only; bf16 and FlashAttention-2 available",
    ),
    "kaggle_t4": Profile(
        name="kaggle_t4",
        arch="7.5",              
        trust_timing=True,
        vram_gb=16.0,
        notes="timing and QLoRA training; no bf16, no FlashAttention (use SDPA)",
    ),
    "kaggle_p100": Profile(
        name="kaggle_p100",
        arch="6.0",               
        trust_timing=True,
        vram_gb=16.0,
        notes="fallback if no T4; fp16 only, much slower",
    ),
}

_BF16_MIN_ARCH = 8.0

def detect_profile() -> Profile:
    override=os.environ.get("HIPKB_PROFILE")
    if override:
        if override not in PROFILES:
            raise ValueError(f"HIPKB_PROFILE={override!r} is not one of {sorted(PROFILES)}")
        return PROFILES[override]
    try:
        import torch
        if not torch.cuda.is_available():
            return PROFILES["laptop"]
        major, minor=torch.cuda.get_device_capability(0)
        arch=f"{major}.{minor}"
        for p in PROFILES.values():
            if p.arch == arch:
                return p
        name=torch.cuda.get_device_name(0)
        total=torch.cuda.get_device_properties(0).total_memory/1e9
        return Profile(name=f"unknown({name})", arch=arch,
                       trust_timing=True, vram_gb=total,
                       notes="auto-detected; not in PROFILES")
    except ImportError:
        return PROFILES["laptop"]
    
def supports_bf16(profile: Profile) -> bool:
    return float(profile.arch) >= _BF16_MIN_ARCH

def hip_cflags(include_dir: Path | None=None) -> list[str]:
    inc = include_dir or HIP_INCLUDE
    return[
        "-D__HIP_PLATFORM_NVIDIA__",
        f"-I{inc}",
        "-O3",
        "--expt-relaxed-constexpr",
        "-Xcompiler", "/Zc:preprocessor",
        "-allow-unsupported-compiler",
    ]
    
HIP_CFLAGS=hip_cflags()

@dataclass(frozen=True)
class EvalConfig:
    n_correctness_trials: int=5
    n_timing_trials: int=100
    n_warmup: int=10
    atol: float=1e-2
    rtol: float=1e-2
    compile_timeout_s: int=180
    run_timeout_s: int=120
    k_samples: int=3
    temperature: float=0.7
    top_p: float=0.95
    max_repair_rounds: int=1
    fast_p_thresholds = (0.0, 0.5, 0.8, 1.0, 1.5, 2.0)
    
EVAL=EvalConfig()

@dataclass(frozen=True)
class RetrievalConfig:
    embed_model: str = "BAAI/bge-large-en-v1.5"  
    embed_dim: int = 1024
    chunk_size: int = 1000
    chunk_overlap: int = 150
    
    top_k_reference: int = 7
    top_k_examples: int = 3
    
    @property
    def top_k_total(self) -> int:
        return self.top_k_reference + self.top_k_examples
    
RETRIEVAL=RetrievalConfig()

@dataclass(frozen=True)
class ModelConfig:
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    max_input_tokens: int = 6144
    max_output_tokens: int = 1536
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
    learning_rate: float = 2e-4
    num_epochs: int = 1
    per_device_batch_size: int = 1
    grad_accum_steps: int = 16
    max_seq_length: int = 2048
    gradient_checkpointing: bool = True
    
MODEL = ModelConfig()

@dataclass
class ConfigReport:
    profile: Profile
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
 
    @property
    def ok(self) -> bool:
        return not self.problems
 
    def render(self) -> str:
        lines = [
            f"profile : {self.profile.name} (sm_{self.profile.arch.replace('.', '')})",
            f"vram : {self.profile.vram_gb:.0f} GB",
            f"timing : {'trusted' if self.profile.trust_timing else 'NOT trusted'}",
            f"bf16 : {'yes' if supports_bf16(self.profile) else 'no'}",
            f"kernelbench: {KERNELBENCH_ROOT}",
            f"hip headers : {HIP_INCLUDE}",
            f"data : {DATA_DIR}",
        ]
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        for p in self.problems:
            lines.append(f"  PROBLEM: {p}")
        return "\n".join(lines)
 
def validate(require_gpu: bool = False) -> ConfigReport:
    """Check the environment is actually usable before doing any work."""
    profile = detect_profile()
    report = ConfigReport(profile=profile)
 
    if not KERNELBENCH_ROOT.is_dir():
        report.problems.append(
            f"KernelBench not found at {KERNELBENCH_ROOT}. "
            "git clone https://github.com/ScalingIntelligence/KernelBench, "
            "or set HIPKB_KERNELBENCH."
        )
 
    if not (HIP_INCLUDE / "hip" / "hip_runtime.h").is_file():
        report.problems.append(
            f"HIP headers not found at {HIP_INCLUDE}. Run scripts/setup_hip.sh, "
            "or set HIPKB_HIP_INCLUDE."
        )
    elif not (HIP_INCLUDE / "hip" / "nvidia_detail").is_dir():
        report.problems.append(
            f"{HIP_INCLUDE}/hip exists but has no nvidia_detail/. The NVIDIA "
            "backend headers live in ROCm/hipother, not ROCm/hip -- rerun "
            "scripts/setup_hip.sh."
        )
    elif not (HIP_INCLUDE / "hip" / "hip_version.h").is_file():
        report.problems.append(
            f"{HIP_INCLUDE}/hip/hip_version.h missing. It is generated by ROCm's "
            "build system and ships in no repo; scripts/setup_hip.sh stubs it."
        )
 
    if require_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                report.problems.append("No CUDA GPU visible but require_gpu=True.")
        except ImportError:
            report.problems.append("torch not installed but require_gpu=True.")
 
    if not profile.trust_timing:
        report.warnings.append(
            f"{profile.name} timings are not trusted; fast_p will be withheld. "
            "Run timing on Kaggle."
        )
    if not profile.can_train_7b:
        report.warnings.append(
            f"{profile.vram_gb:.0f} GB is not enough to QLoRA a 7B model. "
            "Generation and evaluation are fine here; train on Kaggle."
        )
 
    return report
 
if __name__ == "__main__":
    r = validate()
    print(r.render())
    raise SystemExit(0 if r.ok else 1)