import hashlib
import os
import subprocess
from pathlib import Path


def test_phase2_bit_repro(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    cmd = [
        "python3",
        "-m",
        "rkan_hz.rkan_boundary_train",
        "--epochs",
        "5",
        "--batches-per-epoch",
        "2",
        "--batch-size",
        "4",
    ]
    env = dict(os.environ)
    repo_root = Path(__file__).resolve().parents[1]
    monorepo_src = repo_root / "projects" / "rational_kan_hz" / "src"
    standalone_src = repo_root / "src"
    env["PYTHONPATH"] = str(monorepo_src if monorepo_src.is_dir() else standalone_src)
    subprocess.check_call(cmd + ["--out-dir", str(out_a)], env=env)
    subprocess.check_call(cmd + ["--out-dir", str(out_b)], env=env)
    a = (out_a / "final_weights.bin").read_bytes()
    b = (out_b / "final_weights.bin").read_bytes()
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()
