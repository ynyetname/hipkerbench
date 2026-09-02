import os
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompts import (  
    ARMS, PromptSpec, build_prompt, build_repair_prompt,
    extract_model_new, prompt_compliance,
)
from tasks import (  
    Task, assert_disjoint, load_tasks, split_summary, split_tasks,
)

KB_ROOT = Path(os.environ.get("HIPKB_KERNELBENCH", Path.home() / "KernelBench"))
needs_kb = pytest.mark.skipif(not KB_ROOT.is_dir(), reason="KernelBench not checked out")

def _fake_tasks(per_level=(100, 100, 50)) -> list[Task]:
    out = []
    for lv, n in zip((1, 2, 3), per_level):
        for pid in range(1, n + 1):
            out.append(Task(lv, pid, f"Task{pid}", Path(f"/tmp/level{lv}/{pid}_Task{pid}.py")))
    return out

def test_split_is_disjoint():
    train, test = split_tasks(_fake_tasks())
    assert_disjoint(train, test)

def test_split_covers_everything():
    tasks = _fake_tasks()
    train, test = split_tasks(tasks)
    assert len(train) + len(test) == len(tasks)
    assert {t.uid for t in train} | {t.uid for t in test} == {t.uid for t in tasks}

def test_split_is_deterministic_across_calls():
    a_tr, a_te = split_tasks(_fake_tasks())
    b_tr, b_te = split_tasks(_fake_tasks())
    assert [t.uid for t in a_tr] == [t.uid for t in b_tr]
    assert [t.uid for t in a_te] == [t.uid for t in b_te]

def test_split_is_stable_when_tasks_are_added():
    """Adding level-3 tasks must not reshuffle levels 1 and 2.

    Otherwise re-running after a KernelBench update silently changes which
    problems are 'test', and every previous result becomes incomparable.
    """
    small_tr, _ = split_tasks(_fake_tasks((100, 100, 10)))
    big_tr, _ = split_tasks(_fake_tasks((100, 100, 50)))
    l12_small = [t.uid for t in small_tr if t.level in (1, 2)]
    l12_big = [t.uid for t in big_tr if t.level in (1, 2)]
    assert l12_small == l12_big

def test_split_is_stratified_per_level():
    tasks = _fake_tasks()
    train, test = split_tasks(tasks, train_frac=0.6)
    for lv, total in ((1, 100), (2, 100), (3, 50)):
        tr = sum(1 for t in train if t.level == lv)
        assert tr == round(total * 0.6), f"level {lv}: {tr} train, expected {round(total*0.6)}"

def test_different_salt_gives_different_split():
    a, _ = split_tasks(_fake_tasks(), salt="hipkb-v1")
    b, _ = split_tasks(_fake_tasks(), salt="other")
    assert [t.uid for t in a] != [t.uid for t in b]

def test_bad_train_frac_rejected():
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            split_tasks(_fake_tasks(), train_frac=bad)

def test_assert_disjoint_actually_catches_overlap():
    tasks = _fake_tasks((5, 5, 5))
    with pytest.raises(AssertionError, match="contamination"):
        assert_disjoint(tasks, tasks)

@needs_kb
def test_loads_real_kernelbench():
    tasks = load_tasks(KB_ROOT)
    assert len(tasks) == 250, f"expected 250 tasks for levels 1-3, got {len(tasks)}"
    assert sum(1 for t in tasks if t.level == 1) == 100
    assert sum(1 for t in tasks if t.level == 3) == 50

@needs_kb
def test_real_task_source_is_readable():
    tasks = load_tasks(KB_ROOT, levels=(1,))
    relu = next(t for t in tasks if t.name == "ReLU")
    assert "class Model" in relu.source and "def get_inputs" in relu.source

@needs_kb
def test_real_split_summary_renders():
    train, test = split_tasks(load_tasks(KB_ROOT))
    assert_disjoint(train, test)
    assert "level" in split_summary(train, test)

def test_retrieval_arm_refuses_empty_chunks():
    """A silent fallback to no-retrieval would turn the rag arm into the base
    arm and quietly invalidate the comparison."""
    with pytest.raises(ValueError, match="expects retrieved chunks"):
        build_prompt("class Model: pass", ARMS["rag"], retrieved=[])

def test_random_control_requires_retrieval_enabled():
    with pytest.raises(ValueError, match="requires use_retrieval"):
        PromptSpec("bad", use_retrieval=False, random_retrieval=True)

def test_base_arm_has_no_reference_material():
    msgs = build_prompt("class Model: pass", ARMS["base"])
    assert "Reference material" not in msgs[1]["content"]

def test_main_arms_omit_the_portability_hint():
    """Telling the model about warpSize would erase the phenomenon the lint
    exists to measure. Only the dedicated ablation arm may include it."""
    for name in ("base", "rag", "rand", "sft", "rag_sft"):
        spec = ARMS[name]
        msgs = build_prompt("class Model: pass", spec,
                            retrieved=["doc"] if spec.use_retrieval else None)
        assert "warpSize" not in msgs[1]["content"], f"arm '{name}' leaks the hint"
    hint = build_prompt("class Model: pass", ARMS["hint"], retrieved=["doc"])
    assert "warpSize" in hint[1]["content"]

def test_exemplar_uses_cuda_sources_not_cxx_override():
    """Upstream's AMD exemplar sets CXX=hipcc and uses cpp_sources. On the nvcc
    path that fails on every task."""
    from prompts import HIP_NVIDIA_EXEMPLAR
    assert "cuda_sources" in HIP_NVIDIA_EXEMPLAR
    assert "CXX" not in HIP_NVIDIA_EXEMPLAR
    assert "extra_cuda_cflags" in HIP_NVIDIA_EXEMPLAR
    msgs = build_prompt("class Model: pass", ARMS["base"])
    assert 'Do not set os.environ["CXX"]' in msgs[1]["content"]

def test_repair_prompt_preserves_original_context():
    msgs = build_prompt("class Model: pass", ARMS["rag"], retrieved=["HIP doc chunk"])
    repaired = build_repair_prompt(msgs, "broken code", "error: boom")
    assert len(repaired) == len(msgs) + 2
    assert "HIP doc chunk" in repaired[1]["content"], "lost retrieved context on repair"
    assert repaired[-1]["role"] == "user" and "error: boom" in repaired[-1]["content"]

GOOD = """Here you go.

```python
import torch
class ModelNew(torch.nn.Module):
    def forward(self, x):
        return x
```
"""

DRAFT_THEN_FIX = """First attempt:

```python
class ModelNew:
    version = 1
```

Wait, that's wrong. Corrected:

```python
class ModelNew:
    version = 2
```
"""

HELPER_THEN_ANSWER = """```python
def helper():
    return 1
```

```python
class ModelNew:
    pass
```
"""

def test_extracts_simple_block():
    r = extract_model_new(GOOD)
    assert r.ok and "class ModelNew" in r.code and r.code.startswith("import torch")

def test_takes_last_modelnew_when_model_self_corrects():
    r = extract_model_new(DRAFT_THEN_FIX)
    assert "version = 2" in r.code, "must take the corrected block, not the draft"
    assert r.warnings

def test_skips_helper_block_without_modelnew():
    r = extract_model_new(HELPER_THEN_ANSWER)
    assert "class ModelNew" in r.code and "def helper" not in r.code

def test_unfenced_code_still_parses():
    r = extract_model_new("import torch\nclass ModelNew(torch.nn.Module):\n    pass")
    assert r.ok and r.warnings

def test_prose_only_response_fails_cleanly():
    r = extract_model_new("I'm not able to write that kernel.")
    assert not r.ok and "no code block" in r.reason

def test_empty_response_fails_cleanly():
    assert not extract_model_new("").ok
    assert not extract_model_new("   \n  ").ok

def test_block_without_modelnew_reports_why():
    r = extract_model_new("```python\nx = 1\n```")
    assert not r.ok and "no `class ModelNew`" in r.reason and r.fences_found == 1

def test_compliance_flags_cuda_api():
    code = ('import torch\nhip_source = "cudaMalloc(&d, 8);"\n'
            'load_inline(cuda_sources=hip_source)\nclass ModelNew: pass')
    issues = prompt_compliance(code)
    assert any("cuda*" in i for i in issues)
    assert any("hip_runtime.h" in i for i in issues)

def test_compliance_flags_cxx_override():
    code = 'import os\nos.environ["CXX"] = "hipcc"\n#include <hip/hip_runtime.h>\ncuda_sources=1'
    assert any("CXX" in i for i in prompt_compliance(code))

def test_compliance_clean_on_good_kernel():
    from prompts import HIP_NVIDIA_EXEMPLAR
    assert prompt_compliance(HIP_NVIDIA_EXEMPLAR) == []