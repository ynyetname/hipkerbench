REPO_URL = "https://github.com/ynyetname/hipkerbench.git"
ARMS = "base,rag,rand"
LEVELS = "1"
K = 3
LIMIT = None          # set to 5 for a ~15 minute smoke run first

import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / Path(REPO_URL).stem.replace(".git", "")

def sh(cmd, cwd=None, check=True):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {cmd}")
    
# code
if REPO.exists():
    sh("git pull --ff-only", cwd=REPO, check=False)
else:
    sh(f"git clone -q {REPO_URL} {REPO}")

os.chdir(REPO)
sys.path.insert(0, str(REPO))

# dependencies
sh("pip install -q sentence-transformers bitsandbytes peft accelerate")

# KernelBench
KB = WORK / "KernelBench"
if not KB.exists():
    sh(f"git clone -q --depth 1 https://github.com/ScalingIntelligence/KernelBench.git {KB}")
os.environ["HIPKB_KERNELBENCH"] = str(KB)

# Persist outputs under /kaggle/working so they survive to the session's output.
os.environ["HIPKB_DATA"] = str(WORK / "hipkb-data")
os.environ["HIPKB_HIP_INCLUDE"] = str(Path.home() / "hip-nvidia" / "include")

# HIP headers
if not (Path(os.environ["HIPKB_HIP_INCLUDE"]) / "hip" / "hip_runtime.h").is_file():
    sh("python scripts/setup_hip.py")

sh("python config.py")

# corpus and index
data = Path(os.environ["HIPKB_DATA"])
if not (data / "corpus" / "chunks.jsonl").is_file():
    sh("python corpus.py")
if not (data / "index" / "reference.npz").is_file():
    sh("python retrieval.py")

# generation
cmd = f"python generate.py --arms {ARMS} --levels {LEVELS} --k {K}"
if LIMIT:
    cmd += f" --limit {LIMIT}"
sh(cmd)

print("\nGenerations written to:", data / "generations")
for f in sorted((data / "generations").glob("*.jsonl")):
    n = sum(1 for _ in f.open())
    print(f"  {f.name}: {n} records")

print("\nDownload the generations/ folder from the Output panel, "
      "then commit it from your machine.")