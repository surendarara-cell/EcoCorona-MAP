"""Minimal neural regression model for EcoCorona-MAP reproducibility."""

import torch
import torch.nn as nn


class EcoCoronaRegressor(nn.Module):
    """Feed-forward baseline model for adsorption energy prediction.

    This lightweight model is provided as a reproducible template. It can be
    replaced with a full graph neural network implementation when residue-level
    graph features and nanoparticle surface graphs are available.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
