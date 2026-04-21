import json

import pytest

from rkan_hz.rkan_paper_run import _load_trained_evidence, run_target, run_trained_target_sweep, target_specs


def test_paper_run_targets_have_exact_symbolic_extraction_for_in_class(tmp_path):
    in_class = [spec for spec in target_specs() if spec.target_id in {"t1", "t2", "t3", "t4", "t5", "t6"}]
    assert len(in_class) == 6
    for spec in in_class:
        row = run_target(spec, tmp_path)
        assert row["passed"], spec.target_id
        assert row["max_point_residual"] <= row["threshold"]
        assert row["artifact_files"]["pdf_compiled"]


def test_paper_run_out_of_class_rows_report_residual_without_promotion(tmp_path):
    out_of_class = [spec for spec in target_specs() if spec.kind == "out_of_class"]
    assert len(out_of_class) == 2
    for spec in out_of_class:
        row = run_target(spec, tmp_path)
        assert row["rationalization_status"] == "residual_reported"
        assert not row["passed"]
        assert row["boundary_recorded"]
        path = tmp_path / f"{spec.target_id}_convergence.json"
        assert json.loads(path.read_text(encoding="utf-8"))["target_id"] == spec.target_id


def test_paper_run_rows_label_oracle_reconstruction_mode_and_training_not_executed(tmp_path):
    # Post-audit contract: run_target must disclose that it is not training.
    # Failure of this test means the caller may mistake pipeline-identity for
    # training convergence, which is the exact regression the audit caught.
    for spec in target_specs():
        row = run_target(spec, tmp_path)
        assert row["rationalization_mode"] == "oracle_reconstruction", spec.target_id
        assert row["training_executed"] is False, spec.target_id
        # The in-class passing status must be the post-audit label, not the old
        # "exact_match" string which silently implied a trained fit.
        if spec.kind == "in_class" and spec.target_id != "t6":
            assert row["rationalization_status"] == "oracle_reconstruction_exact", spec.target_id


def test_trained_target_sweep_emits_real_training_evidence(tmp_path):
    pytest.importorskip("torch")
    spec = next(spec for spec in target_specs() if spec.target_id == "t2")
    summary = run_trained_target_sweep(
        spec,
        tmp_path / "trained" / spec.target_id,
        devices=["cpu"],
        samples=1,
        train_points=256,
        val_points=256,
        steps=400,
        lr=0.05,
        denominator_bound=1_000_000,
    )
    assert summary["target_id"] == "t2"
    assert summary["passed"]
    assert summary["exact_match_all_runs"]
    assert summary["semantic_faithfulness_passed"]
    loaded = _load_trained_evidence(tmp_path, "t2")
    assert loaded is not None
    assert loaded["rationalization_passed"]
    assert loaded["identity_passed"]
