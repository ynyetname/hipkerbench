import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate as G  # noqa: E402
from errors import CompileError, RuntimeError_  # noqa: E402
from kbench import KernelResult, prepare_source  # noqa: E402
from lint import lint_kernel  # noqa: E402
from prompts import ARMS  # noqa: E402
from tasks import Task  # noqa: E402

GOOD_RESPONSE = "```python\nclass ModelNew:\n    pass\n```"

class StubLM:
    """Satisfies the LanguageModel Protocol. Records what it was asked."""

    def __init__(self, responses=None):
        self.calls = 0
        self.seen_messages: list[list[dict]] = []
        self.seen_seeds: list[int] = []
        self._responses = responses

    def chat(self, messages, seed=0, **kw):
        self.calls += 1
        self.seen_messages.append(messages)
        self.seen_seeds.append(seed)
        if self._responses:
            return self._responses[min(self.calls - 1, len(self._responses) - 1)]
        return GOOD_RESPONSE

def _task(problem_id=19, level=1, tmp=None) -> Task:
    tmp = tmp or Path("/tmp/hipkb_tasks")
    d = tmp / f"level{level}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{problem_id}_T{problem_id}.py"
    p.write_text("import torch\n"
                 "class Model(torch.nn.Module):\n"
                 "    def forward(self, x):\n        return torch.relu(x)\n")
    return Task(level, problem_id, f"T{problem_id}", p)

def _result(compiled, correct, *, err="", compile_error=None,
            runtime_error=None, lint_src=None) -> KernelResult:
    return KernelResult(
        task_uid="L1_P19", arm="base",
        compiled=compiled, correct=correct, raw_error=err,
        compile_error=compile_error, runtime_error=runtime_error,
        lint=lint_kernel(lint_src) if lint_src else None,
    )

def _script(monkeypatch, results):
    """Make evaluate_kernel return a scripted sequence, one per round."""
    calls = {"n": 0}

    def fake(**kw):
        r = results[min(calls["n"], len(results) - 1)]
        calls["n"] += 1
        return r

    monkeypatch.setattr(G, "evaluate_kernel", fake)
    return calls

def test_repair_fixes_a_failure_on_the_second_round(monkeypatch):
    _script(monkeypatch, [
        _result(False, False, err='error: identifier "hipFoo" is undefined',
                compile_error=CompileError.UNKNOWN_HIP_API),
        _result(True, True),
    ])
    lm = StubLM()
    rec = G.generate_one(_task(), ARMS["base"], lm, max_repair_rounds=1)

    assert [a.round for a in rec.attempts] == [0, 1]
    assert not rec.attempts[0].correct and rec.attempts[1].correct
    assert rec.succeeded
    assert lm.calls == 2

def test_zero_repair_rounds_reproduces_the_paper(monkeypatch):
    """max_repair_rounds=0 is the paper's setting and the control the repair
    arm is measured against. It must be expressible."""
    _script(monkeypatch, [
        _result(False, False, err="error: expected a ';'",
                compile_error=CompileError.SYNTAX),
        _result(True, True),
    ])
    lm = StubLM()
    rec = G.generate_one(_task(), ARMS["base"], lm, max_repair_rounds=0)

    assert len(rec.attempts) == 1, "no repair round may be attempted"
    assert lm.calls == 1
    assert not rec.succeeded

def test_success_on_round_zero_skips_repair(monkeypatch):
    _script(monkeypatch, [_result(True, True)])
    lm = StubLM()
    rec = G.generate_one(_task(), ARMS["base"], lm, max_repair_rounds=1)
    assert len(rec.attempts) == 1 and lm.calls == 1

def test_unrepairable_failure_does_not_burn_a_generation(monkeypatch):
    """An OOM is not the model's fault. Retrying one wastes budget."""
    _script(monkeypatch, [
        _result(True, False, err="HIP out of memory. Tried to allocate 20 GiB",
                runtime_error=RuntimeError_.OOM),
        _result(True, True),
    ])
    lm = StubLM()
    rec = G.generate_one(_task(), ARMS["base"], lm, max_repair_rounds=1)
    assert len(rec.attempts) == 1
    assert lm.calls == 1, "must not retry an unrepairable failure"

def test_repair_prompt_carries_the_compiler_error(monkeypatch):
    _script(monkeypatch, [
        _result(False, False, err='error: identifier "hipBogus" is undefined',
                compile_error=CompileError.UNKNOWN_HIP_API),
        _result(True, True),
    ])
    lm = StubLM()
    G.generate_one(_task(), ARMS["base"], lm, max_repair_rounds=1)

    repair_prompt = lm.seen_messages[1][-1]["content"]
    assert "hipBogus" in repair_prompt

def test_repair_prompt_never_shows_the_injected_preamble(monkeypatch):
    """Showing our own boilerplate back to the model would teach it to emit it,
    and then prepare_source would find a marker it did not write."""
    prepared = prepare_source("class ModelNew:\n    pass\n")
    _script(monkeypatch, [
        _result(False, False, err="error: expected a ';'",
                compile_error=CompileError.SYNTAX),
        _result(True, True),
    ])
    lm = StubLM(responses=[f"```python\n{prepared}```", GOOD_RESPONSE])
    G.generate_one(_task(), ARMS["base"], lm, max_repair_rounds=1)

    shown_back = lm.seen_messages[1][-2]["content"]
    assert "HIP_CFLAGS = [" not in shown_back

def test_unparseable_response_counts_as_a_failure(monkeypatch):
    _script(monkeypatch, [_result(True, True)])
    lm = StubLM(responses=["I'm not able to write that kernel."])
    rec = G.generate_one(_task(), ARMS["base"], lm, max_repair_rounds=0)

    assert not rec.succeeded
    assert rec.attempts[0].code is None
    assert not rec.attempts[0].compiled, "a response we cannot parse never compiled"

def test_parse_failure_does_not_call_the_evaluator(monkeypatch):
    calls = _script(monkeypatch, [_result(True, True)])
    lm = StubLM(responses=["no code here"])
    G.generate_one(_task(), ARMS["base"], lm, max_repair_rounds=0)
    assert calls["n"] == 0

def test_retrieval_arm_refuses_to_run_without_a_retriever(monkeypatch):
    """Silently running `rag` without retrieval would turn it into `base` and
    quietly invalidate the whole comparison."""
    _script(monkeypatch, [_result(True, True)])
    with pytest.raises(ValueError):
        G.generate_one(_task(), ARMS["rag"], StubLM(), retrieve=None)

def test_base_arm_needs_no_retriever(monkeypatch):
    _script(monkeypatch, [_result(True, True)])
    rec = G.generate_one(_task(), ARMS["base"], StubLM(), retrieve=None)
    assert rec.succeeded

def test_retrieved_context_reaches_the_prompt(monkeypatch):
    _script(monkeypatch, [_result(True, True)])
    lm = StubLM()
    G.generate_one(_task(), ARMS["rag"], lm,
                   retrieve=lambda task, spec: ["CHUNK_MARKER_XYZ"])
    assert "CHUNK_MARKER_XYZ" in lm.seen_messages[0][1]["content"]

def test_seed_derived_from_identity_not_a_counter(monkeypatch):
    """A global counter would change every task's seed as soon as you reorder
    the list or resume from a different point."""
    _script(monkeypatch, [_result(True, True)])
    a = StubLM(); b = StubLM()
    G.generate_one(_task(19), ARMS["base"], a, sample_idx=2, max_repair_rounds=0)
    G.generate_one(_task(19), ARMS["base"], b, sample_idx=2, max_repair_rounds=0)
    assert a.seen_seeds == b.seen_seeds

def test_different_samples_get_different_seeds(monkeypatch):
    _script(monkeypatch, [_result(True, True)])
    lm = StubLM()
    for s in (0, 1, 2):
        G.generate_one(_task(19), ARMS["base"], lm, sample_idx=s, max_repair_rounds=0)
    assert len(set(lm.seen_seeds)) == 3

def test_different_arms_get_different_seeds(monkeypatch):
    _script(monkeypatch, [_result(True, True)])
    lm = StubLM()
    G.generate_one(_task(19), ARMS["base"], lm, max_repair_rounds=0)
    G.generate_one(_task(19), ARMS["sft"], lm, max_repair_rounds=0)
    assert lm.seen_seeds[0] != lm.seen_seeds[1]

def test_rerun_makes_no_model_calls(monkeypatch, tmp_path):
    """A 12-hour session limit means a script that loses 180 completed tasks
    because it died on 181 is unusable."""
    _script(monkeypatch, [_result(True, True)])
    tasks = [_task(19, tmp=tmp_path), _task(20, tmp=tmp_path)]
    lm = StubLM()

    G.run_arm(tasks, "base", lm, k_samples=2, out_dir=tmp_path,
              max_repair_rounds=0, verbose=False)
    after_first = lm.calls
    assert after_first == 4

    G.run_arm(tasks, "base", lm, k_samples=2, out_dir=tmp_path,
              max_repair_rounds=0, verbose=False)
    assert lm.calls == after_first, "re-run must skip everything already on disk"


def test_rerun_does_not_duplicate_records(monkeypatch, tmp_path):
    _script(monkeypatch, [_result(True, True)])
    tasks = [_task(19, tmp=tmp_path)]
    lm = StubLM()
    p = G.run_arm(tasks, "base", lm, k_samples=2, out_dir=tmp_path,
                  max_repair_rounds=0, verbose=False)
    G.run_arm(tasks, "base", lm, k_samples=2, out_dir=tmp_path,
              max_repair_rounds=0, verbose=False)
    assert len(G.load_records(p)) == 2


def test_truncated_final_line_survives(monkeypatch, tmp_path):
    """A hard kill leaves a half-written line. That must not poison the file."""
    _script(monkeypatch, [_result(True, True)])
    tasks = [_task(19, tmp=tmp_path)]
    lm = StubLM()
    p = G.run_arm(tasks, "base", lm, k_samples=1, out_dir=tmp_path,
                  max_repair_rounds=0, verbose=False)

    with p.open("a", encoding="utf-8") as f:
        f.write('{"task_uid": "L1_P99", "arm": "ba')

    assert len(G.load_records(p)) == 1
    assert len(G.completed_keys(p)) == 1


def test_resume_after_truncation_regenerates_only_the_lost_record(monkeypatch, tmp_path):
    _script(monkeypatch, [_result(True, True)])
    tasks = [_task(19, tmp=tmp_path), _task(20, tmp=tmp_path)]
    lm = StubLM()
    p = G.run_arm(tasks, "base", lm, k_samples=1, out_dir=tmp_path,
                  max_repair_rounds=0, verbose=False)

    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text(lines[0] + "\n" + lines[1][:30], encoding="utf-8")

    before = lm.calls
    G.run_arm(tasks, "base", lm, k_samples=1, out_dir=tmp_path,
              max_repair_rounds=0, verbose=False)
    assert lm.calls - before == 1, "exactly the lost record should be redone"


def test_missing_generations_file_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate.py"):
        G.load_records(tmp_path / "nope.jsonl")

def test_every_attempt_is_kept_not_just_the_last(monkeypatch):
    """Extension #4 trains on (broken + error) -> fixed. The intermediate
    attempts ARE the training data; the paper discarded them."""
    _script(monkeypatch, [
        _result(False, False, err='error: identifier "hipX" is undefined',
                compile_error=CompileError.UNKNOWN_HIP_API),
        _result(True, True),
    ])
    rec = G.generate_one(_task(), ARMS["base"], StubLM(), max_repair_rounds=1)
    assert len(rec.attempts) == 2
    assert rec.attempts[0].code and rec.attempts[0].error
    assert rec.attempts[1].code


def test_record_json_round_trip(monkeypatch):
    _script(monkeypatch, [_result(True, True)])
    rec = G.generate_one(_task(), ARMS["base"], StubLM(), max_repair_rounds=0)
    back = G.GenerationRecord.from_json(rec.to_json())
    assert back.task_uid == rec.task_uid
    assert back.succeeded == rec.succeeded
    assert len(back.attempts) == len(rec.attempts)


def test_wavefront_failure_is_attributed_to_the_platform(monkeypatch):
    """The lint feeding through into the record is what keeps AMD-inherited
    wave-64 bugs out of the model-error bucket."""
    wave64 = ("#include <hip/hip_runtime.h>\n"
              "__global__ void k(float* v) {\n"
              "    for (int o = 64 / 2; o > 0; o >>= 1)\n"
              "        v[0] += __shfl_down(v[0], o, 64);\n}\n")
    _script(monkeypatch, [_result(True, False, lint_src=wave64)])
    rec = G.generate_one(_task(), ARMS["base"], StubLM(), max_repair_rounds=0)
    assert rec.attempts[0].attribution == "platform_artifact"


def test_summarize_runs_on_real_records(monkeypatch, tmp_path):
    _script(monkeypatch, [_result(True, True)])
    tasks = [_task(19, tmp=tmp_path), _task(20, tmp=tmp_path)]
    lm = StubLM()
    p = G.run_arm(tasks, "base", lm, k_samples=2, out_dir=tmp_path,
                  max_repair_rounds=0, verbose=False)
    out = G.summarize(G.load_records(p))
    assert "level" in out and "1" in out