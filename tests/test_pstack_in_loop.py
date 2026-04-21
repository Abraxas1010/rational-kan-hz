from __future__ import annotations

import json
from pathlib import Path
from fractions import Fraction

from rkan_hz.rkan_pstack_training import (
    PStackAccumulatorWorker,
    RationalKANDeg8,
    assert_bit_identical,
    backward_pstack,
    benchmark_inference,
    benchmark_microbatch_sweep,
    benchmark_update_lane,
    forward_pstack,
    run,
    sb_grid_batch,
    target_degree8,
)


def test_forward_bit_identical_fraction_vs_pstack():
    worker = PStackAccumulatorWorker.instance()
    for seed in (20260420, 20260421, 20260422):
        model = RationalKANDeg8.seeded(seed=seed)
        batch = sb_grid_batch(6, seed)
        dense = model.forward_fraction(batch)
        lazy = forward_pstack(model, batch, worker)
        assert dense["predictions"] == lazy["predictions"]
        assert dense["pre_activations"] == lazy["pre_activations"]
        assert dense["activations"] == lazy["activations"]
        assert dense["edge_outputs"] == lazy["edge_outputs"]


def test_backward_bit_identical_fraction_vs_pstack():
    worker = PStackAccumulatorWorker.instance()
    for seed in (20260420, 20260421, 20260422):
        model = RationalKANDeg8.seeded(seed=seed)
        batch = sb_grid_batch(6, seed)
        targets = [target_degree8(x) for x in batch]
        dense_forward = model.forward_fraction(batch)
        lazy_forward = forward_pstack(model, batch, worker)
        dense = model.backward_fraction(batch, targets, dense_forward)
        lazy = backward_pstack(model, batch, targets, lazy_forward, worker)
        assert dense["loss"] == lazy["loss"]
        assert dense["output_weight_grads"] == lazy["output_weight_grads"]
        assert dense["bias_grad"] == lazy["bias_grad"]
        assert dense["edge_grads"] == lazy["edge_grads"]


def test_training_step_bit_identical():
    result = assert_bit_identical(steps=2, batch_size=4, seed=20260420)
    assert result["parity_gate"] is True


def test_factored_sum_groups_matches_fraction_on_mixed_denominators():
    worker = PStackAccumulatorWorker.instance()
    groups = [
        [Fraction(1, 2), Fraction(1, 3), Fraction(-5, 6), Fraction(7, 30)],
        [Fraction(11, 64), Fraction(-3, 64), Fraction(5, 8), Fraction(9, 10)],
        [Fraction(0), Fraction(0), Fraction(4, 9)],
    ]
    values, stats = worker.sum_groups(groups)
    assert values == [sum(group, Fraction(0)) for group in groups]
    assert stats["distinct_denominators"] >= 3


def test_network_meets_scale_requirement():
    model = RationalKANDeg8.seeded(seed=20260420)
    spec = model.network_spec()
    assert spec["degree"] >= 8
    assert spec["hidden"] >= 10
    assert spec["meets_scale_requirement"] is True


def test_update_speed_benchmark_schema():
    model = RationalKANDeg8.seeded(seed=20260420)
    result = benchmark_update_lane(model, samples=2, batch_size=4, seed=20260420, microbatches=4)
    assert result["benchmark_kind"] == "amortized_training_update"
    assert result["samples"] == 2
    assert len(result["rows"]) == 2
    assert result["rows"][0]["output_hash_dense"] == result["rows"][0]["output_hash_pstack"]
    assert result["rows"][0]["pstack_distinct_denominators"] > 0


def test_inference_benchmark_schema():
    model = RationalKANDeg8.seeded(seed=20260420)
    result = benchmark_inference(model, samples=1, batch_size=2, seed=20260420)
    assert result["benchmark_kind"] == "network_inference"
    assert result["samples"] == 1
    assert result["rows"][0]["output_hash_dense"] == result["rows"][0]["output_hash_pstack"]


def test_microbatch_sweep_schema():
    model = RationalKANDeg8.seeded(seed=20260420)
    result = benchmark_microbatch_sweep(
        model,
        samples=1,
        batch_size=4,
        seed=20260420,
        microbatch_grid=(1, 4),
    )
    assert result["benchmark_kind"] == "update_microbatch_amortization_sweep"
    assert result["grid"] == [1, 4]
    assert len(result["rows"]) == 2
    assert all(row["hash_parity_passed"] for row in result["rows"])


def test_run_emits_artifacts(tmp_path: Path):
    out_dir = tmp_path / "pstack_in_loop"
    result = run(
        out_dir,
        samples=1,
        steps=1,
        batch_size=2,
        devices=("cpu",),
        seed=20260420,
        skip_speed_benchmark=False,
        microbatches=4,
        inference_samples=1,
        microbatch_sweep_samples=1,
        microbatch_grid=(1, 4),
    )
    assert (out_dir / "bitwise_parity_log.jsonl").exists()
    assert (out_dir / "network_spec.json").exists()
    assert (out_dir / "inference_speedup.json").exists()
    assert (out_dir / "update_speedup.json").exists()
    assert (out_dir / "microbatch_amortization_sweep.json").exists()
    assert (out_dir / "report.md").exists()
    update_speed = json.loads((out_dir / "update_speedup.json").read_text(encoding="utf-8"))
    inference_speed = json.loads((out_dir / "inference_speedup.json").read_text(encoding="utf-8"))
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert update_speed["cpu"]["benchmark_kind"] == "amortized_training_update"
    assert inference_speed["cpu"]["benchmark_kind"] == "network_inference"
    assert summary["speed"]["update"]["cpu"]["benchmark_kind"] == "amortized_training_update"
    assert summary["speed"]["inference"]["cpu"]["benchmark_kind"] == "network_inference"
    assert result["parity_gate"] is True
    assert result["scale_gate"] is True
