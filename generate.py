from __future__ import annotations
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol
 
from config import EVAL, GENERATIONS_DIR, MODEL
from errors import truncate_for_prompt
from kbench import KernelResult, evaluate_kernel, strip_preamble
from prompts import ARMS, PromptSpec, build_prompt, build_repair_prompt, extract_model_new
from tasks import Task

@dataclass
class Attempt:
    """One generation and what the toolchain said about it."""
    round: int
    response: int
    code: str | None=None
    parse_note: str=""
    compiled: bool=False
    correct: bool=False
    error: str=""
    error_label: str=""
    attribution: str=""
    
    compliance: list[str] = field(default_factory=list)
    seconds: float = 0.0
 
    @property
    def ok(self) -> bool:
        return self.compiled and self.correct
    
@dataclass
class GenerationRecord:
    """Everything about one (task, arm, sample): prompt, context, every attempt."""
    task_uid: str
    arm: str
    sample_idx: int
    level: int
    attempts: list[Attempt] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    model: str = ""
 
    @property
    def key(self) -> str:
        """Identity for resume. One record per task/arm/sample."""
        return f"{self.task_uid}|{self.arm}|{self.sample_idx}"
 
    @property
    def succeeded(self) -> bool:
        return any(a.ok for a in self.attempts)
 
    @property
    def solved_first_try(self) -> bool:
        return bool(self.attempts) and self.attempts[0].ok
 
    @property
    def repair_helped(self) -> bool:
        """Failed, then succeeded after seeing a compiler error.
 
        This is the headline number for the repair arm, and each of these is
        also a training example for file 11.
        """
        return (len(self.attempts) > 1
                and not self.attempts[0].ok
                and self.attempts[-1].ok)
 
    @property
    def final(self) -> Attempt | None:
        return self.attempts[-1] if self.attempts else None
 
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
 
    @classmethod
    def from_json(cls, line: str) -> "GenerationRecord":
        d = json.loads(line)
        d["attempts"] = [Attempt(**a) for a in d.get("attempts", [])]
        return cls(**d)

class LanguageModel(Protocol):
    """A Protocol so the whole loop is testable with a stub -- no 7B download,
    no GPU. Same reason retrieval.py has an Embedder Protocol."""
 
    name: str
 
    def chat(self, messages: list[dict[str, str]], *, temperature: float,
             max_new_tokens: int, seed: int | None = None) -> str: ...
 
class HFModel:
    """Qwen2.5-Coder-7B-Instruct in 4-bit"""
 
    def __init__(self, model_name: str | None = None, device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
 
        from config import detect_profile, supports_bf16
 
        self.name = model_name or MODEL.base_model
        profile = detect_profile()
        dtype = torch.bfloat16 if supports_bf16(profile) else torch.float16
 
        quant = None
        if MODEL.load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
 
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            quantization_config=quant,
            dtype=dtype,
            device_map=device_map,
            # Turing (T4) has no FlashAttention; SDPA works everywhere Ampere-.
            attn_implementation="sdpa",
        )
        self.model.eval()
        self._torch = torch
 
    def chat(self, messages, *, temperature, max_new_tokens, seed=None) -> str:
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)
 
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=MODEL.max_input_tokens,
        ).to(self.model.device)
 
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=EVAL.top_p if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Slice off the prompt so we return only what was generated.
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
 
    def unload(self) -> None:
        del self.model
        self._torch.cuda.empty_cache()

RetrieveFn = Callable[[Task, PromptSpec], list[str]]
 
def generate_one(
    task: Task,
    spec: PromptSpec,
    model: LanguageModel,
    *,
    sample_idx: int = 0,
    retrieve: RetrieveFn | None = None,
    max_repair_rounds: int | None = None,
    device: int = 0,
) -> GenerationRecord:
    """Generate, compile, verify, and repair one kernel.
 
    Repair stops early on an unrepairable failure -- an OOM or a timeout is not
    the model's fault and retrying one burns generation budget for nothing.
    """
    rounds = EVAL.max_repair_rounds if max_repair_rounds is None else max_repair_rounds
 
    retrieved: list[str] = []
    if spec.use_retrieval:
        if retrieve is None:
            raise ValueError(
                f"arm '{spec.name}' needs retrieval but no retrieve function was "
                "given. Silently running it without context would turn it into "
                "the base arm and invalidate the comparison."
            )
        retrieved = retrieve(task, spec)
 
    record = GenerationRecord(
        task_uid=task.uid, arm=spec.name, sample_idx=sample_idx,
        level=task.level, retrieved=retrieved, model=getattr(model, "name", "?"),
    )
 
    messages = build_prompt(task.source, spec, retrieved or None)
    # Seed from task, arm and sample so k samples differ from each other but the
    # whole run reproduces.
    seed = abs(hash(f"{task.uid}|{spec.name}|{sample_idx}")) % (2**31)
 
    for rnd in range(rounds + 1):
        started = time.time()
        response = model.chat(
            messages,
            temperature=EVAL.temperature if EVAL.k_samples > 1 else 0.0,
            max_new_tokens=MODEL.max_new_tokens,
            seed=seed + rnd,
        )
        attempt = Attempt(round=rnd, response=response,
                          seconds=round(time.time() - started, 2))
 
        parsed = extract_model_new(response)
        if not parsed.ok:
            # A response we cannot parse is a compile failure -- the honest
            # treatment, since nothing runnable was produced.
            attempt.parse_note = parsed.reason
            attempt.error = f"could not extract ModelNew: {parsed.reason}"
            attempt.error_label = "parse_failure"
            attempt.attribution = "compile_failure"
            record.attempts.append(attempt)
            break
 
        attempt.code = parsed.code
 
        result: KernelResult = evaluate_kernel(
            task_uid=task.uid, arm=spec.name,
            reference_source=task.source, model_source=parsed.code,
            sample_idx=sample_idx, repair_round=rnd,
            measure_time=False,          # timing is file 10's job
            device=device,
        )
 
        attempt.compiled = result.compiled
        attempt.correct = result.correct
        attempt.error = result.error_for_prompt()
        attempt.error_label = (result.compile_error or result.runtime_error or "").__str__()
        attempt.attribution = result.attribution
        attempt.compliance = result.compliance_issues
        record.attempts.append(attempt)
 
        if result.correct:
            break
        if rnd >= rounds:
            break
        if result.metadata.get("retryable"):
            continue
        if not result.repairable:
            break
 
        messages = build_repair_prompt(
            messages,
            failed_source=strip_preamble(parsed.code),
            error_text=truncate_for_prompt(attempt.error),
        )
 
    return record
 
def run_arm(
    tasks: Iterable[Task],
    arm: str,
    model: LanguageModel,
    *,
    retrieve: RetrieveFn | None = None,
    k_samples: int | None = None,
    out_dir: Path | None = None,
    max_repair_rounds: int | None = None,
    device: int = 0,
    verbose: bool = True,
) -> Path:
    """Run one arm over tasks, appending to a JSONL file. Safe to re-run.
 
    Records are appended one line at a time and flushed immediately, so a killed
    session loses at most the task in flight.
    """
    spec = ARMS[arm] if isinstance(arm, str) else arm
    k = EVAL.k_samples if k_samples is None else k_samples
    out_dir = out_dir or GENERATIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.name}.jsonl"
 
    done = completed_keys(path)
    if verbose and done:
        print(f"[{spec.name}] resuming: {len(done)} records already on disk")
 
    tasks = list(tasks)
    todo = [(t, i) for t in tasks for i in range(k)
            if f"{t.uid}|{spec.name}|{i}" not in done]
 
    if verbose:
        print(f"[{spec.name}] {len(todo)} of {len(tasks) * k} to generate")
 
    with path.open("a", encoding="utf-8") as f:
        for n, (task, sample_idx) in enumerate(todo, start=1):
            record = generate_one(
                task, spec, model, sample_idx=sample_idx, retrieve=retrieve,
                max_repair_rounds=max_repair_rounds, device=device,
            )
            f.write(record.to_json() + "\n")
            f.flush()                    # survive a hard kill
            if verbose:
                mark = "ok " if record.succeeded else "   "
                extra = " (repaired)" if record.repair_helped else ""
                print(f"  [{n}/{len(todo)}] {mark}{task.uid} s{sample_idx}"
                      f" r{len(record.attempts) - 1}{extra}")
 
    return path
 
def completed_keys(path: Path) -> set[str]:
    """Which task/arm/sample triples are already on disk.
 
    Tolerates a truncated final line, which is what a killed session leaves.
    """
    if not path.is_file():
        return set()
    keys: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(GenerationRecord.from_json(line).key)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
    return keys
 
def load_records(path: Path) -> list[GenerationRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"No generations at {path}. Run generate.py first.")
    out: list[GenerationRecord] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(GenerationRecord.from_json(line))
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue
    return out
 
def summarize(records: list[GenerationRecord]) -> str:
    """Per-level breakdown for one arm.
 
    pass@1 is the paper's setting (one greedy sample). pass@k is what k samples
    buys, and reporting both is extension #8 -- a single sample tells you very
    little about a high-variance task.
    """
    by_level: dict[int, list[GenerationRecord]] = {}
    for r in records:
        by_level.setdefault(r.level, []).append(r)
 
    lines = [f"{'level':<7}{'tasks':>7}{'pass@1':>9}{'pass@k':>9}{'repaired':>10}"]
    for level, group in sorted(by_level.items()):
        tasks = {r.task_uid for r in group}
        first = [r for r in group if r.sample_idx == 0]
        p1 = sum(1 for r in first if r.succeeded) / max(len(first), 1)
        pk = sum(1 for uid in tasks
                 if any(r.succeeded for r in group if r.task_uid == uid)) / max(len(tasks), 1)
        rep = sum(1 for r in group if r.repair_helped)
        lines.append(f"{level:<7}{len(tasks):>7}{p1:>8.1%}{pk:>9.1%}{rep:>10}")
    return "\n".join(lines)
 
def error_profile(records: list[GenerationRecord]) -> dict[str, int]:
    """Failure buckets for the final attempt of each record.
 
    This is the breakdown that makes the retrieval claim falsifiable: RAG should
    specifically shrink unknown_hip_api. If correctness improves without that
    bucket shrinking, the gain came from context length, not documentation.
    """
    counts: dict[str, int] = {}
    for r in records:
        a = r.final
        if a is None or a.ok:
            continue
        counts[a.error_label or "unknown"] = counts.get(a.error_label or "unknown", 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))