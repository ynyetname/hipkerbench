import importlib
import os
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402

def _reload(monkeypatch, **env):
    """Reload config with a patched environment, since paths resolve at import."""
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    return importlib.reload(cfg)

def test_hip_cflags_respects_custom_include():
    flags = cfg.hip_cflags(Path("/custom/inc"))
    assert "-I/custom/inc" in flags
    assert "-D__HIP_PLATFORM_NVIDIA__" in flags

def test_laptop_cannot_train_7b_but_kaggle_can():
    assert not cfg.PROFILES["laptop"].can_train_7b
    assert cfg.PROFILES["kaggle_t4"].can_train_7b


def test_bf16_support_matches_architecture():
    assert cfg.supports_bf16(cfg.PROFILES["laptop"])        # Ampere 8.6
    assert not cfg.supports_bf16(cfg.PROFILES["kaggle_t4"])  # Turing 7.5
    assert not cfg.supports_bf16(cfg.PROFILES["kaggle_p100"])  # Pascal 6.0


def test_profile_override_works(monkeypatch):
    monkeypatch.setenv("HIPKB_PROFILE", "kaggle_t4")
    assert cfg.detect_profile().name == "kaggle_t4"


def test_unknown_profile_override_fails_loudly(monkeypatch):
    monkeypatch.setenv("HIPKB_PROFILE", "gaming_rig")
    with pytest.raises(ValueError, match="not one of"):
        cfg.detect_profile()

def test_repair_can_be_disabled_to_match_the_paper():
    """max_repair_rounds=0 must be expressible, since that is the paper's
    setting and the control our repair arm is measured against."""
    import dataclasses
    baseline = dataclasses.replace(cfg.EVAL, max_repair_rounds=0)
    assert baseline.max_repair_rounds == 0
    assert cfg.EVAL.max_repair_rounds >= 1  # our default differs deliberately

def test_validate_reports_missing_kernelbench(monkeypatch, tmp_path):
    m = _reload(monkeypatch,
                HIPKB_KERNELBENCH=tmp_path / "nope",
                HIPKB_HIP_INCLUDE=tmp_path / "inc",
                HIPKB_DATA=tmp_path / "data")
    r = m.validate()
    assert not r.ok
    assert any("KernelBench not found" in p for p in r.problems)

def test_validate_detects_missing_nvidia_detail(monkeypatch, tmp_path):
    """The single most likely setup mistake: cloning ROCm/hip and stopping,
    without realising the NVIDIA headers live in a different repo."""
    inc = tmp_path / "inc" / "hip"
    inc.mkdir(parents=True)
    (inc / "hip_runtime.h").write_text("// stub")
    m = _reload(monkeypatch,
                HIPKB_KERNELBENCH=tmp_path,
                HIPKB_HIP_INCLUDE=tmp_path / "inc",
                HIPKB_DATA=tmp_path / "data")
    r = m.validate()
    assert any("nvidia_detail" in p and "hipother" in p for p in r.problems)

def test_validate_detects_missing_hip_version(monkeypatch, tmp_path):
    inc = tmp_path / "inc" / "hip"
    (inc / "nvidia_detail").mkdir(parents=True)
    (inc / "hip_runtime.h").write_text("// stub")
    m = _reload(monkeypatch,
                HIPKB_KERNELBENCH=tmp_path,
                HIPKB_HIP_INCLUDE=tmp_path / "inc",
                HIPKB_DATA=tmp_path / "data")
    r = m.validate()
    assert any("hip_version.h" in p for p in r.problems)

def test_validate_passes_on_a_complete_setup(monkeypatch, tmp_path):
    inc = tmp_path / "inc" / "hip"
    (inc / "nvidia_detail").mkdir(parents=True)
    (inc / "hip_runtime.h").write_text("// stub")
    (inc / "hip_version.h").write_text("// stub")
    m = _reload(monkeypatch,
                HIPKB_KERNELBENCH=tmp_path,
                HIPKB_HIP_INCLUDE=tmp_path / "inc",
                HIPKB_DATA=tmp_path / "data",
                HIPKB_PROFILE="kaggle_t4")
    r = m.validate()
    assert r.ok, r.problems
    assert "trusted" in r.render()

def test_laptop_setup_is_ok_but_warns_about_timing(monkeypatch, tmp_path):
    inc = tmp_path / "inc" / "hip"
    (inc / "nvidia_detail").mkdir(parents=True)
    (inc / "hip_runtime.h").write_text("// stub")
    (inc / "hip_version.h").write_text("// stub")
    m = _reload(monkeypatch,
                HIPKB_KERNELBENCH=tmp_path,
                HIPKB_HIP_INCLUDE=tmp_path / "inc",
                HIPKB_DATA=tmp_path / "data",
                HIPKB_PROFILE="laptop")
    r = m.validate()
    assert r.ok, "a laptop is a valid machine for compile + correctness"
    assert any("not trusted" in w for w in r.warnings)
    assert any("7B" in w for w in r.warnings)

def test_ensure_dirs_creates_everything(monkeypatch, tmp_path):
    m = _reload(monkeypatch, HIPKB_DATA=tmp_path / "d")
    m.ensure_dirs()
    for d in (m.CORPUS_DIR, m.INDEX_DIR, m.GENERATIONS_DIR,
              m.RESULTS_DIR, m.SFT_DIR, m.BUILD_DIR):
        assert d.is_dir(), d

@pytest.fixture(autouse=True)
def _restore_config():
    """Reloads in these tests mutate module state; put it back afterwards."""
    yield
    for k in ("HIPKB_KERNELBENCH", "HIPKB_HIP_INCLUDE", "HIPKB_DATA", "HIPKB_PROFILE"):
        import os
        os.environ.pop(k, None)
    importlib.reload(cfg)