#!/usr/bin/env python3
"""
Cognitive Lorenz Field Analysis
===============================
Visualizes the dynamics of the Cognitive Lorenz Field.
"""

import torch
import matplotlib.pyplot as plt
from cognitive_lorenz import CognitiveLorenzField

def run_analysis():
    print("Running Cognitive Lorenz Field Analysis...")
    dim = 64
    torch.manual_seed(42)

    vL0 = torch.randn(dim)
    vJ0 = torch.randn(dim)

    model = CognitiveLorenzField(dim=dim)
    output = model(vL0, vJ0, steps=1000)

    traj_L = output["vL_traj"]
    traj_J = output["vJ_traj"]
    traj_z = output["lorenz_z"].cpu().numpy()

    distances = torch.norm(traj_L - traj_J, dim=1).cpu().numpy()
    steps = range(len(distances))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Distance between vL and vJ
    ax1.plot(steps, distances, 'b-', label='Embedding Distance ||vL - vJ||')
    ax1.set_title("Cognitive Convergence over Time")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Distance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Lorenz Z coordinate (Modulation factor)
    ax2.plot(steps, traj_z, 'r-', label='Lorenz Z (Coupling Modulation)')
    ax2.set_title("Lorenz Attractor Modulation Dynamics")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Z amplitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lorenz_field_analysis.png")
    print("Analysis complete. Visualization saved to lorenz_field_analysis.png")

if __name__ == "__main__":
    run_analysis()
