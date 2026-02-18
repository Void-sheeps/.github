#!/usr/bin/env python3
"""
Phi Bilateral Infinite Line Field (Rebuilt)
==========================================

Line: (-∞) ---- 0 ---- (+∞)
A → direita
B → esquerda

Metric: phi-projection
Integration: from P^0 = 0 to P^f = f

This module:
- Generates bilateral contraction field
- Compares P^k in Ascension(A) vs Regression(B)
- Measures symmetry
- Computes informational perturbation
"""

import torch
import torch.nn as nn
import math
import argparse


class PhiInfiniteLine(nn.Module):

    def __init__(self, max_k=64, learnable_phi=False, device="cpu"):
        super().__init__()
        self.device = device
        self.max_k = max_k

        phi_value = (1 + math.sqrt(5)) / 2

        if learnable_phi:
            self.phi = nn.Parameter(torch.tensor(phi_value))
        else:
            self.register_buffer("phi", torch.tensor(phi_value))

    def projection(self, f, k):
        return f * torch.pow(self.phi, -k)

    def forward(self, f):
        """
        f : scalar endpoint reference (Pf = f)
        """

        f = torch.tensor(f, dtype=torch.float32, device=self.device)

        k = torch.arange(0, self.max_k, device=self.device, dtype=torch.float32)

        # φ-regulated projection
        proj = self.projection(f, k)

        # Bilateral branches
        A_branch = -proj
        B_branch = +proj

        # Distance symmetry
        distance = B_branch - A_branch  # = 2 * proj

        # Symmetry verification
        symmetry_error = torch.abs(torch.abs(A_branch) - torch.abs(B_branch))

        # Informational density (natural index compression)
        density = k / (distance + 1e-12)

        # Extract requested points
        P2_A  = A_branch[2]
        P2_B  = B_branch[2]

        P32_A = A_branch[32] if self.max_k > 32 else None
        P32_B = B_branch[32] if self.max_k > 32 else None

        return {
            "k": k,
            "A_branch": A_branch,
            "B_branch": B_branch,
            "distance": distance,
            "density": density,
            "symmetry_error": symmetry_error,
            "P2_A": P2_A,
            "P2_B": P2_B,
            "P32_A": P32_A,
            "P32_B": P32_B
        }


def run_simulation():
    model = PhiInfiniteLine(max_k=64)
    output = model(f=1.0)

    print("\n--- Phi Infinite Line Field Simulation ---")
    print("\n--- P² Comparison ---")
    print("Ascension A:", output["P2_A"].item())
    print("Regression B:", output["P2_B"].item())

    print("\n--- P³² Comparison ---")
    print("Ascension A:", output["P32_A"].item())
    print("Regression B:", output["P32_B"].item())

    print("\nMax symmetry error:",
          torch.max(output["symmetry_error"]).item())
    print("\nSimulation Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phi Bilateral Infinite Line Field Simulation")
    parser.add_argument("--simulate", action="store_true", help="Run the field simulation")
    args = parser.parse_args()

    if args.simulate or not any(vars(args).values()):
        run_simulation()
