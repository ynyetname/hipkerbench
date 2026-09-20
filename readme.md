# HIP Kernel Generation on NVIDIA

An investigation into whether retrieval over HIP/ROCm documentation and
supervised fine-tuning on synthetic data actually improve an LLM's ability to
write correct, performant GPU kernels, measured on KernelBench and run
end-to-end on NVIDIA hardware via HIP's CUDA backend.

**Background.** This project builds on and substantially revises the design
of *Adapting Language Models for Low-Resource GPU Kernel Programming*
(Konidala, Pahlavan & Antony, Stanford CS224N), which asked the same question
on AMD MI350X. Three parts of that design don't support the conclusions drawn
from them — an evaluation split that wasn't actually held out, an ablation
grid too small to separate its two interventions, and a failure-mode analysis
done by hand — so this project fixes those, adds two extensions the original
never tried, and works through three consequences of moving to NVIDIA
hardware. Expect lower numbers here, and treat them as not comparable to the
original's.

## Approach

**Platform.** HIP compiled against the **NVIDIA backend** — header-only, no
ROCm install. `hip_runtime.h` routes to `nvidia_detail/` when
`__HIP_PLATFORM_NVIDIA__` is defined, and nvcc compiles the result as ordinary
CUDA. Verified working on a Kaggle T4.

**Held-out evaluation.** The train/test split happens before any generation:
deterministic, stratified, 60/40, so no synthetic training kernel is
generated from a task that later shows up in evaluation.

**Complete ablation grid.** Three arms can't decompose two interventions
(retrieval, fine-tuning). Adds an SFT-only arm and, more importantly, a
**random-chunk retrieval control** — if random chunks perform like relevant
ones, "retrieval helps" is really "long context helps".

**Automatic error taxonomy.** Classifying failures automatically rather than
by hand makes the claim falsifiable: RAG should specifically shrink the
`unknown_hip_api` bucket, because that's what documentation fixes.

**Compiler-in-the-loop repair.** HIP's compiler is close to a perfect
verifier, so it's worth using live, not just offline to filter training data.
Feeding one compiler error back at inference costs almost nothing and is the
baseline any fine-tuning arm has to beat.

**Training on repair trajectories.** Examples are built as *(broken kernel +
compiler error) → fixed kernel*, not just finished code, so training teaches
the model to use compiler feedback rather than imitate an endpoint.

**Portability lint.** AMD wavefronts are 64 threads, NVIDIA warps are 32.
Kernels inheriting a hardcoded 64 from retrieved AMD docs compile cleanly and
return wrong answers. Those are flagged statically and attributed as
`platform_artifact` rather than `model_error`, so retrieval arms aren't
penalised for the corpus doing its job.

**Corpus split by document type.** Retrieved *kernel examples* push the model
toward over-aggressive optimisation while *reference prose* helps — mixing
both into one index hides that. Indexed separately here, so the ratio is a
knob (7 reference + 3 examples, summing to 10 retrieved chunks).

**k=3 sampling.** Single greedy generations give no measure of variance in a
high-variance task.

## Experimental arms

| Arm | Retrieval | SFT | Repair | Purpose |
|---|---|---|---|---|
| `base` | — | — | — | no retrieval, no fine-tuning |
| `rag` | relevant | — | — | retrieval alone |
| `rand` | **random** | — | — | control: is it retrieval or context length? |
| `sft` | — | yes | — | fine-tuning's independent contribution |
| `rag_sft` | relevant | yes | — | retrieval + fine-tuning combined |
| `repair` | relevant | — | 1 round | does the compiler beat fine-tuning? |
| `hint` | relevant | — | — | ablation: does warning about wave-64 help? |

## Setup

### Local (Windows/Linux/macOS) — writing code, corpus, tests

```bash
conda create -n hipkerbench python=3.11 -y
conda activate hipkerbench
pip install -r requirements.txt

git clone https://github.com/ScalingIntelligence/KernelBench.git
python scripts/setup_hip.py          # fetch HIP headers (3 repos, pinned)
python config.py                     # should report no PROBLEMs
python -m pytest tests/ -q
```

Compiling kernels locally also needs nvcc and a C++ compiler. On Windows that
means CUDA Toolkit 12.x matching your PyTorch build, plus Visual Studio Build
Tools **2022** — nvcc 12.6 does not support VS 2026 and `cudafe++` crashes.
If you hit that, skip local compilation and use Kaggle; the laptop was only
ever doing compile checks anyway.

### Kaggle — everything that needs a GPU

Accelerator: GPU T4 x2. Internet: On.

```python
!git clone https://github.com/<you>/<repo>.git && cd <repo>
!pip install -q -r requirements.txt
!python scripts/setup_hip.py
!python corpus.py && python retrieval.py
!python generate.py --arms base,rag,rand --levels 1 --k 3
```

## Workflow

```
write in VS Code  →  push to GitHub  →  pull on Kaggle  →  run  →  download results
```

Nothing important lives on Kaggle; it is disposable compute. Results come
back as files you commit from your machine.

## Run order

| Step | Command | GPU | Time |
|---|---|---|---|
| 1 | `python scripts/setup_hip.py` | no | 1 min |
| 2 | `python corpus.py` | no | 2 min |
| 3 | `python retrieval.py` | helps | 10 min (1.3 GB download) |
| 4 | `python generate.py --arms ... --levels 1` | **yes** | see below |
| 5 | `python evaluate.py` | **yes** | timing pass |
| 6 | `python sft_data.py` | no | 1 min |
| 7 | `python train_sft.py` | **yes** | ~2 h |
| 8 | `python report.py` | no | seconds |

### GPU budget

| Scope | Generations | GPU-hours | Kaggle quota |
|---|---|---|---|
| Level 1 only | 720 | ~13 | half a week |
| Levels 1+2 | 1,440 | ~26 | ~1 week |
| All levels | 1,800 | ~32 | just over 1 week |

Start with Level 1. You get a complete result across all arms in half a week
and find out whether the pipeline holds before committing a month to it.

Generation is resumable — records are appended and flushed one at a time, so
a session killed at the 12-hour limit loses at most the task in flight.

## Layout

```
config.py       paths, hardware profiles, all constants, design invariants
tasks.py        KernelBench loading + deterministic train/test split
prompts.py      per-arm prompt construction, response parsing
lint.py         static AMD/NVIDIA portability lint
errors.py       compile and runtime failure taxonomy
kbench.py       KernelBench adapter for the HIP-on-NVIDIA path
corpus.py       fetch + chunk HIP docs into reference/example collections
retrieval.py    bge-large embedding, two-collection search, random control
generate.py     generation driver with compiler repair, resumable
evaluate.py     timing pass and fast_p                      (pending)
sft_data.py     build the repair-trajectory training set    (pending)
train_sft.py    QLoRA                                       (pending)
report.py       aggregate into tables                       (pending)

scripts/        setup_hip.py, smoke_test.py, kaggle_smoke_test.py
tests/          98 tests, none requiring a GPU
```

## Design notes

**Compile here, time there.** `config.PROFILES` marks laptop timings as
untrusted, and `KernelResult.speedup` returns `None` unless timing came from
a machine flagged otherwise. A thermally throttled laptop number cannot reach
a fast_p table by any route.

**Everything is testable without a GPU.** The `Embedder` and `LanguageModel`
Protocols exist so the retrieval and generation logic can be exercised with
stubs. 98 tests run in under a second on any machine.

**Version pinning.** HIP headers are pinned to `rocm-7.0.2` — the newest
release that still targets CUDA 12. ROCm 7.1+ calls `cudaMemcpyBatchAsync`, a
CUDA 13 API. The corpus is pinned to the same tag so retrieved documentation
describes the API the kernels actually compile against.

## Known limitations

- Speedups are T4-relative and **not comparable** to the MI350X numbers in
  the original paper.
- Correctness on real AMD hardware would differ; the lint estimates that gap
  but does not measure it.
- Corpus is 198 documents against the original's 590 — they scraped the
  rendered site including generated API pages, this takes the source tree.
- Retrieval uses exact cosine search in numpy rather than ChromaDB. At 3,369
  chunks ChromaDB does exact search too, so results are equivalent.