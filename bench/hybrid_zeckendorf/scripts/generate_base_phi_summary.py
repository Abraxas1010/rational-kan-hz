#!/usr/bin/env python3
import json
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_result(repo_root: Path, name: str) -> Path:
    candidates = [
        repo_root / "bench" / "hybrid_zeckendorf" / "results" / f"{name}.json",
        repo_root / "results" / f"{name}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(name)


def main() -> int:
    script = Path(__file__).resolve()
    repo_root = script.parent.parent.parent.parent
    result_names = [
        "exp8_base_phi_rawmul",
        "exp9_base_phi_crossover",
        "exp10_shift_bridge",
    ]
    results = {name: load_json(resolve_result(repo_root, name)) for name in result_names}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = (
        repo_root / "artifacts" / "ops" / "zeckendorf_papers_base_phi" / ts
    )
    archive_dir.mkdir(parents=True, exist_ok=True)

    exp8 = results["exp8_base_phi_rawmul"]
    exp9 = results["exp9_base_phi_crossover"]
    exp10 = results["exp10_shift_bridge"]

    summary = {
        "experiment_suite": "zeckendorf_papers_base_phi_exact_tranche",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "formal_backing": {
            "theorems": [
                "rawBasePhiMul",
                "basePhiEval_rawBasePhiMul",
                "shiftToBasePhi_semantics",
                "basePhiEval_mul",
            ],
            "lean_module": "lean/HeytingLean/Bridge/Veselov/HybridZeckendorf/BasePhi.lean",
        },
        "experiments": {
            name: {
                "decision": results[name].get("decision"),
                "summary": results[name].get("summary", {}),
                "data_points": len(results[name].get("data_points", [])),
            }
            for name in result_names
        },
        "decision": {
            "exp8_sparse_vs_repeated": exp8.get("decision"),
            "exp9_sparse_dense_crossover_rho": exp9.get("summary", {}).get("crossover_rho"),
            "exp10_shift_bridge": exp10.get("decision"),
            "exact_lane_status": "useful_on_sparse_base_phi_digits",
            "integration_boundary": "Not yet integrated into HybridNumber multiplication; utility confirmed only for the exact base-phi/shift-support tranche.",
        },
    }

    summary_path = archive_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for name in result_names:
        src = resolve_result(repo_root, name)
        dst = archive_dir / src.name
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
