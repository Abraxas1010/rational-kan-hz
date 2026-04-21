#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


paper = load("artifacts/rational_kan_hz/paper_run/paper_run_summary.json")
scaling = load("artifacts/rational_kan_hz/paper_run/scaling/speedup_vs_features.json")
pstack = load("artifacts/rational_kan_hz/paper_run/pstack_in_loop/summary.json")
gpu = load("artifacts/rational_kan_hz/gpu_scaled_training_iter_1/gpu_scaled_training_summary.json")

assert scaling["passed"] is True, "scaling alpha gate failed"
assert pstack["parity_gate"] is True, "pstack parity gate failed"
assert pstack["scale_gate"] is True, "pstack scale gate failed"
assert gpu["identity_passed"] is True, "gpu identity gate failed"
assert gpu["rationalization_passed"] is True, "gpu rationalization gate failed"
assert gpu["convergence_passed"] is True, "gpu convergence gate failed"
assert gpu["speed_gate_training_ci_lower_gt_1"] is True, "gpu training speed gate failed"
assert gpu["passed"] is True, "gpu summary failed"

print("paper_commit:", paper["commit"])
print("scaling_alpha:", scaling["alpha"])
print("pstack_speed_gate:", pstack["speedup_gate"])
print("gpu_training_speed_mean:", gpu["dense_over_p4_training_speedup"]["mean"])
print("gpu_memory_ratio_mean:", gpu["p4_over_dense_memory_ratio"]["mean"])
