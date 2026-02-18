#!/usr/bin/env python3
"""
Phi Bilateral Infinite Line Field Analysis
==========================================
Generates visualizations for the phi-projection field.
"""

import torch
import matplotlib.pyplot as plt
from phi_infinite_line import PhiInfiniteLine

def run_analysis():
    print("Running Phi Infinite Line Analysis...")
    model = PhiInfiniteLine(max_k=64)
    output = model(f=1.0)

    k = output["k"].cpu().numpy()
    A = output["A_branch"].cpu().numpy()
    B = output["B_branch"].cpu().numpy()
    density = output["density"].cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Bilateral Branches
    ax1.plot(k, A, 'r-', label='Ascension (A)')
    ax1.plot(k, B, 'b-', label='Regression (B)')
    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax1.set_title("Phi-Projection Bilateral Branches")
    ax1.set_xlabel("k (Index)")
    ax1.set_ylabel("Position P^k")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Informational Density
    ax2.plot(k, density, 'g-', label='Density (k/distance)')
    ax2.set_title("Informational Density (Compression)")
    ax2.set_xlabel("k (Index)")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("phi_line_analysis.png")
    print("Analysis complete. Visualization saved to phi_line_analysis.png")

if __name__ == "__main__":
    run_analysis()
