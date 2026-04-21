"""Phase 0 PyTorch reference baseline for rational KAN.

This module reconstructs a small Kolmogorov-Arnold network from the notebook's
degree-3 rational activation. It deliberately imports no Boundary or HZ code:
the output is a numerical reference only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


SEED = 42
TARGET_FN_SOURCE = "x[..., 0] ** 2 + torch.sin(x[..., 1])"
EXPECTED_TARGET_FN_SOURCE_HASH = (
    "d328b8f1b771866f67c2894620557b71a54576c7466200520cc3392501a7b810"
)
TARGET_MSE = 0.1


def repo_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def set_determinism(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


class RationalActivation(nn.Module):
    """Degree-3-over-degree-3 rational activation from the source notebook."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerator = self.a[0] + self.a[1] * x + self.a[2] * x**2 + self.a[3] * x**3
        denominator = 1 + self.b[0] * x + self.b[1] * x**2 + self.b[2] * x**3
        return numerator / denominator


class RationalKAN(nn.Module):
    """KAN reconstruction with 2 inputs, 5 outer units, and 10 inner edges."""

    def __init__(self, input_dim: int = 2, outer_count: int = 5) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.outer_count = outer_count
        self.inner = nn.ModuleList(
            nn.ModuleList(RationalActivation() for _ in range(input_dim))
            for _ in range(outer_count)
        )
        self.outer = nn.ModuleList(RationalActivation() for _ in range(outer_count))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outer_outputs: list[torch.Tensor] = []
        for k in range(self.outer_count):
            inner_sum = sum(self.inner[k][i](x[..., i]) for i in range(self.input_dim))
            outer_outputs.append(self.outer[k](inner_sum))
        return sum(outer_outputs)


def target_fn(x: torch.Tensor) -> torch.Tensor:
    return x[..., 0] ** 2 + torch.sin(x[..., 1])


def target_fn_source_hash() -> str:
    return hashlib.sha256(TARGET_FN_SOURCE.encode("utf-8")).hexdigest()


def make_dataset(seed: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x_all = torch.rand((samples, 2), generator=generator, dtype=torch.float64) * 4 - 2
    y_all = target_fn(x_all)
    return x_all, y_all


def state_dict_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode("utf-8"))
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class TrainAttempt:
    lr: float
    epochs: int
    batch_size: int
    final_mse: float
    validation_mse: float
    passed: bool
    weights_hash: str
    seconds: float


def full_mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(nn.functional.mse_loss(model(x), y).item())


def train_once(
    *,
    seed: int,
    lr: float,
    epochs: int,
    batch_size: int,
    samples: int,
    validation_samples: int,
) -> tuple[RationalKAN, list[float], TrainAttempt]:
    set_determinism(seed)
    model = RationalKAN().to(dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x_all, y_all = make_dataset(seed, samples)
    x_val, y_val = make_dataset(seed + 10_000, validation_samples)
    loss_curve: list[float] = []
    start = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    for _epoch in range(epochs):
        idx = torch.randperm(samples, generator=generator)[:batch_size]
        xb = x_all[idx]
        yb = y_all[idx]
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xb)
        loss = nn.functional.mse_loss(prediction, yb)
        loss.backward()
        optimizer.step()
        loss_curve.append(float(loss.item()))
    seconds = time.perf_counter() - start
    final_mse = full_mse(model, x_all, y_all)
    validation_mse = full_mse(model, x_val, y_val)
    attempt = TrainAttempt(
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        final_mse=final_mse,
        validation_mse=validation_mse,
        passed=final_mse <= TARGET_MSE,
        weights_hash=state_dict_hash(model),
        seconds=seconds,
    )
    return model, loss_curve, attempt


def parse_lr_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def choose_best(attempts: Iterable[tuple[RationalKAN, list[float], TrainAttempt]]):
    return min(attempts, key=lambda row: row[2].final_mse)


def write_loss_plot(curve: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve, linewidth=1.2)
    ax.set_title("Phase 0 RKAN Baseline Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("batch MSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_status(result: dict, path: Path) -> None:
    header = "PROMOTED" if result["passed"] else "CLOSED-WITH-FINDINGS"
    lines = [
        f"# {header}",
        "",
        f"Final full-dataset MSE: {result['final_mse']:.12g}",
        f"Validation MSE: {result['validation_mse']:.12g}",
        f"Target MSE: {result['target_mse']}",
        f"Seed: {result['seed']}",
        f"Commit: {result['commit']}",
        f"Target function hash: {result['target_fn_source_hash']}",
        "",
        "Interpretation: Phase 0 is a PyTorch numerical reference only. It is not imported by Boundary or HZ phases.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/rational_kan_hz/iter_1/phase0_baseline")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--validation-samples", type=int, default=1000)
    parser.add_argument("--lr-list", default="0.01,0.005,0.001")
    args = parser.parse_args(argv)

    observed_hash = target_fn_source_hash()
    if observed_hash != EXPECTED_TARGET_FN_SOURCE_HASH:
        raise SystemExit(
            f"target hash drift: {observed_hash} != {EXPECTED_TARGET_FN_SOURCE_HASH}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[tuple[RationalKAN, list[float], TrainAttempt]] = []
    for lr in parse_lr_list(args.lr_list):
        attempts.append(
            train_once(
                seed=args.seed,
                lr=lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                samples=args.samples,
                validation_samples=args.validation_samples,
            )
        )
    best_model, best_curve, best_attempt = choose_best(attempts)
    weights_path = output_dir / "weights.pt"
    torch.save(best_model.state_dict(), weights_path)
    (output_dir / "loss_curve.json").write_text(
        json.dumps(best_curve, indent=2) + "\n",
        encoding="utf-8",
    )
    write_loss_plot(best_curve, output_dir / "loss_curve.png")

    result = {
        "phase": "0",
        "seed": args.seed,
        "architecture": {
            "name": "RationalKAN",
            "input_dim": 2,
            "outer_count": 5,
            "inner_edges_total": 10,
            "activation_degree": 3,
        },
        "target_fn_source": TARGET_FN_SOURCE,
        "target_fn_source_hash": observed_hash,
        "expected_target_fn_source_hash": EXPECTED_TARGET_FN_SOURCE_HASH,
        "final_mse": best_attempt.final_mse,
        "validation_mse": best_attempt.validation_mse,
        "target_mse": TARGET_MSE,
        "passed": best_attempt.passed,
        "commit": repo_commit(),
        "python": sys.version,
        "torch_version": torch.__version__,
        "best_attempt": asdict(best_attempt),
        "attempts": [asdict(attempt) for _model, _curve, attempt in attempts],
        "weights_path": str(weights_path),
        "weights_hash": best_attempt.weights_hash,
        "notes": "PyTorch reference only. Boundary/HZ phases must not import this module.",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_status(result, output_dir / "PHASE0_STATUS.md")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
