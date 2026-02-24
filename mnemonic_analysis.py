#!/usr/bin/env python3
"""
Mnemonic Tracer Analysis
========================
Generates visualizations for the mnemonic projection field.
"""

import torch
import matplotlib.pyplot as plt
from mnemonic_tracer import MnemonicTracer, MNEMONICS, build_n2_matrix

def run_analysis():
    print("Running Mnemonic Tracer Analysis...")

    indices = []
    projections = []

    # Analyze all mnemonics
    for idx, mnem in enumerate(MNEMONICS, start=1):
        matrix = build_n2_matrix(idx)
        proj = matrix.sum().item()

        indices.append(idx)
        projections.append(proj)

    plt.figure(figsize=(12, 6))
    plt.plot(indices, projections, 'g-', label='N^2 Projection Value')
    plt.fill_between(indices, projections, color='green', alpha=0.1)

    plt.title("Mnemonic Projection Field Magnitude (N^2 expansion)")
    plt.xlabel("Mnemonic Index (N)")
    plt.ylabel("Projection Value (sum(matrix))")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("mnemonic_analysis.png")
    print("Analysis complete. Visualization saved to mnemonic_analysis.png")

if __name__ == "__main__":
    run_analysis()
