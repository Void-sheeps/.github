#!/usr/bin/env python3
"""
Relativistic Omega Analysis
===========================
Simulates iterative interaction between two batches of states and
visualizes the convergence of reinforcement weights.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import numpy as np
from relativistic_omega import RelativisticOmegaOperator

def run_simulation(iterations=500, L=8, K=16, batch_size=4):
    print(f"--- Relativistic Omega Operator Simulation (L={L}, K={K}) ---")
    torch.manual_seed(42)

    # Initialize Operator
    model = RelativisticOmegaOperator(L, K)

    # Initialize two batches of states (Simplex normalized)
    Si = F.softmax(torch.randn(batch_size, L, K), dim=-1)
    Sj = F.softmax(torch.randn(batch_size, L, K), dim=-1)

    agreement_history = []
    w_history = []

    print(f"Running interaction for {iterations} iterations...")
    for i in range(iterations):
        # 1. Update Si based on Sj interaction
        Si_new = model(Si, Sj)

        # 2. Update reinforcement based on agreement between new and old state
        model.update_reinforcement(Si, Si_new)

        # 3. Track metrics
        # Mean dot product as a proxy for agreement/stability
        agreement = (Si * Si_new).sum(dim=-1).mean().item()
        agreement_history.append(agreement)

        # Track evolution of weights W
        w_history.append(model.W.detach().clone().numpy())

        # Advance state
        Si = Si_new.detach()

    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # 1. Agreement History
    ax1.plot(agreement_history, color='tab:blue')
    ax1.set_title("State Agreement Metric over Time")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Mean Agreement")
    ax1.grid(True, alpha=0.3)

    # 2. Weight Evolution
    w_history = np.array(w_history)
    for k in range(K):
        ax2.plot(w_history[:, k], alpha=0.6, label=f'W[{k}]' if K <= 8 else "")
    ax2.set_title(f"Reinforcement Weights (W) Evolution (K={K})")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Weight Value")
    ax2.grid(True, alpha=0.3)
    if K <= 8:
        ax2.legend(loc='upper right', fontsize='small', ncol=2)

    plt.tight_layout()
    plt.savefig("relativistic_analysis.png")
    print("\nVisualization saved to relativistic_analysis.png")

    # Final Summary
    print("\nFinal Reinforcement Weights (W):")
    print(model.W.detach().numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relativistic Omega Analysis")
    parser.add_argument("--simulate", action="store_true", help="Run simulation")
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()

    if args.simulate or True:
        run_simulation(iterations=args.iterations)
