"""Rational activation extracted from the Forex_Rational_Graph_Network notebook.

The notebook contains several copies of the same degree-3-over-degree-3
activation. This file preserves the PyTorch form used in the GCN cells. It is a
reference artifact for Phase 0 and is not imported by Boundary/HZ phases.
"""

import torch
import torch.nn as nn


class RationalActivation(nn.Module):
    """Trainable rational activation from the source notebook."""

    def __init__(self) -> None:
        super().__init__()
        self.a0 = nn.Parameter(torch.tensor(0.0))
        self.a1 = nn.Parameter(torch.tensor(1.0))
        self.a2 = nn.Parameter(torch.tensor(0.0))
        self.a3 = nn.Parameter(torch.tensor(0.0))
        self.b1 = nn.Parameter(torch.tensor(0.0))
        self.b2 = nn.Parameter(torch.tensor(0.0))
        self.b3 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerator = self.a0 + self.a1 * x + self.a2 * x**2 + self.a3 * x**3
        denominator = 1 + self.b1 * x + self.b2 * x**2 + self.b3 * x**3
        return numerator / denominator
