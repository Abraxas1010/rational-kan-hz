#!/usr/bin/env python3
import json
from datetime import datetime, timezone
from pathlib import Path

CONJECTURE_ID = "hybrid_zeckendorf_computational_bench_20260307"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    script = Path(__file__).resolve()
    bench_root = script.parent.parent
    repo_root = bench_root.parent.parent
    results_dir = bench_root / "results"
    repo_results_dir = repo_root / "results"

    def resolve_result(name: str) -> Path:
        bench_path = results_dir / f"{name}.json"
        repo_path = repo_results_dir / f"{name}.json"
        if bench_path.exists():
            return bench_path
        return repo_path

    result_paths = {
        "exp1_modexp": resolve_result("exp1_modexp"),
        "exp2_polymul": resolve_result("exp2_polymul"),
        "exp3_density": resolve_result("exp3_density"),
        "exp4_sparse_add": resolve_result("exp4_sparse_add"),
        "exp5_lazy_accum": resolve_result("exp5_lazy_accum"),
        "exp6_sparse_modexp": resolve_result("exp6_sparse_modexp"),
        "exp7_crossover": resolve_result("exp7_crossover"),
        "exp8_base_phi_rawmul": resolve_result("exp8_base_phi_rawmul"),
        "exp9_base_phi_crossover": resolve_result("exp9_base_phi_crossover"),
        "exp10_shift_bridge": resolve_result("exp10_shift_bridge"),
        "exp11_production_multiply": resolve_result("exp11_production_multiply"),
        "exp12_production_dispatch": resolve_result("exp12_production_dispatch"),
        "exp13_production_number_multiply": resolve_result("exp13_production_number_multiply"),
        "exp14_base_phi_bridge_niche": resolve_result("exp14_base_phi_bridge_niche"),
    }
    loaded = {
        name: load_json(path) for name, path in result_paths.items() if path.exists()
    }

    if "exp1_modexp" not in loaded or "exp3_density" not in loaded:
        raise SystemExit(
            "Missing core result files: exp1_modexp.json and exp3_density.json are required"
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = repo_root / "artifacts" / "ops" / "hybrid_zeckendorf_bench" / ts
    archive_dir.mkdir(parents=True, exist_ok=True)

    def status_for(name: str) -> str:
        data = loaded.get(name)
        if data is None:
            return "missing"
        if name == "exp2_polymul" and not data.get("data_points"):
            return "blocked"
        return "completed"

    experiments = {}
    for name in result_paths:
        data = loaded.get(name)
        experiments[name] = {
            "status": status_for(name),
            "decision": data.get("decision", "unknown") if data else "missing",
            "summary": data.get("summary", {}) if data else {},
            "data_points": len(data.get("data_points", [])) if data else 0,
        }

    summary = {
        "experiment_suite": "hybrid_zeckendorf_computational_bench",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conjecture_id": CONJECTURE_ID,
        "experiments": experiments,
        "sparse_regime_extension": {
            "implemented": all(name in loaded for name in [
                "exp4_sparse_add",
                "exp5_lazy_accum",
                "exp6_sparse_modexp",
                "exp7_crossover",
            ]),
            "exp4_table_1e6": loaded.get("exp4_sparse_add", {})
            .get("summary", {})
            .get("table_at_decision_bits", []),
            "exp5_key_ratio": loaded.get("exp5_lazy_accum", {})
            .get("summary", {})
            .get("key_ratio_count100_rho1e3_lazy_vs_gmp"),
            "exp7_crossover_rho": loaded.get("exp7_crossover", {})
            .get("summary", {})
            .get("crossover_rho_lazy"),
        },
        "formal_backing": {
            "theorems_exercised": [
                "multiplyBinary_correct",
                "add_correct",
                "normalize_sound",
                "normalize_canonical",
                "normalize_is_closure_operator",
                "canonical_eval_injective",
                "active_levels_bound",
                "density_upper_bound",
                "supportCard_single_level_bound",
                "weight_closed",
                "carryAt_preserves_eval",
                "add_comm_repr",
            ],
            "formalization_path": "lean/HeytingLean/Bridge/Veselov/HybridZeckendorf/",
        },
        "exact_base_phi_extension": {
            "implemented": all(
                name in loaded
                for name in [
                    "exp8_base_phi_rawmul",
                    "exp9_base_phi_crossover",
                    "exp10_shift_bridge",
                ]
            ),
            "exp8_decision": loaded.get("exp8_base_phi_rawmul", {})
            .get("decision"),
            "exp9_crossover_rho": loaded.get("exp9_base_phi_crossover", {})
            .get("summary", {})
            .get("crossover_rho"),
            "exp10_decision": loaded.get("exp10_shift_bridge", {})
            .get("decision"),
            "theorems_exercised": [
                "rawBasePhiMul",
                "basePhiEval_rawBasePhiMul",
                "shiftToBasePhi_semantics",
                "basePhiEval_mul",
            ],
        },
        "production_multiply_extension": {
            "implemented": "exp11_production_multiply" in loaded,
            "decision": loaded.get("exp11_production_multiply", {})
            .get("decision"),
            "sweep_rows": loaded.get("exp11_production_multiply", {})
            .get("summary", {})
            .get("sweep_rows", []),
            "surface": "HybridNumber::multiply",
        },
        "production_dispatch_extension": {
            "implemented": "exp12_production_dispatch" in loaded,
            "decision": loaded.get("exp12_production_dispatch", {})
            .get("decision"),
            "summary": loaded.get("exp12_production_dispatch", {})
            .get("summary", {}),
            "surface": "HybridNumber::multiply_production",
        },
        "production_number_extension": {
            "implemented": "exp13_production_number_multiply" in loaded,
            "decision": loaded.get("exp13_production_number_multiply", {})
            .get("decision"),
            "summary": loaded.get("exp13_production_number_multiply", {})
            .get("summary", {}),
            "surface": "ProductionNumber::multiply",
        },
        "production_base_phi_niche_extension": {
            "implemented": "exp14_base_phi_bridge_niche" in loaded,
            "decision": loaded.get("exp14_base_phi_bridge_niche", {})
            .get("decision"),
            "summary": loaded.get("exp14_base_phi_bridge_niche", {})
            .get("summary", {}),
            "surface": "HybridNumber::multiply_production_deferred (curated bridge domain)",
        },
    }

    summary_path = archive_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for path in result_paths.values():
        if path.exists():
            target = archive_dir / path.name
            target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
