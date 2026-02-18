#!/usr/bin/env python3
"""
Phi Convergence Layer Analysis
==============================
Visualizes the gating curve of the learned Phi Convergence Layer.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from phi_convergence import PhiConvergenceLayer

def run_analysis():
    print("Running Phi Convergence Layer Analysis...")
    feature_dim = 64

    # Initialize model
    model = PhiConvergenceLayer(feature_dim, init_phi=1.618)

    # Simulate a "trained" state similar to the script output
    # Final Phi was around 1.533
    model.log_phi.data = torch.log(torch.tensor(1.533630))
    model.f.data = torch.tensor(1.0490)

    # Get metrics
    gate_curve, final_phi = model.get_field_metrics()

    k = model.k.detach().numpy()
    gate = gate_curve.detach().numpy()

    # Also calculate intermediate values for plotting
    phi = torch.exp(model.log_phi)
    projection = model.f * torch.pow(phi, -model.k)
    width = 2.0 * torch.abs(projection) + model.epsilon
    density = model.k / width

    proj_np = projection.detach().numpy()
    density_np = density.detach().numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Projection and Density
    ax1.plot(k, proj_np, 'r-', label='Projection P(k)')
    ax1.set_ylabel('Amplitude', color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(k, density_np, 'g-', label='Density D(k)')
    ax1_twin.set_ylabel('Density', color='g')
    ax1_twin.tick_params(axis='y', labelcolor='g')

    ax1.set_title(f"Phi Convergence Field (Phi={final_phi.item():.4f})")
    ax1.grid(True, alpha=0.3)

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # Plot 2: Final Gating Curve
    ax2.plot(k, gate, 'b-', label='Gating Signal (tanh(D))')
    ax2.set_title("Learned Feature Modulation Mask")
    ax2.set_xlabel("Feature Index (k)")
    ax2.set_ylabel("Weight")
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("phi_convergence_analysis.png")
    print("Analysis complete. Visualization saved to phi_convergence_analysis.png")

if __name__ == "__main__":
    run_analysis()
