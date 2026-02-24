#!/usr/bin/env python3
"""
Mnemonic Tracer Analysis
========================
Generates visualizations for the mnemonic projection field.
"""

import torch
import matplotlib.pyplot as plt
from mnemonic_tracer import MnemonicTracer, MNEMONICS

def run_analysis():
    print("Running Mnemonic Tracer Analysis...")
    tracer = MnemonicTracer(MNEMONICS)

    indices = []
    projections = []

    # Analyze all mnemonics
    for idx, mnem in enumerate(MNEMONICS, start=1):
        state = tracer.build_state(idx)
        reflected = tracer.reflect(state)
        expanded = tracer.expand(reflected)
        proj = tracer.project(expanded)

        indices.append(idx)
        projections.append(proj)

    plt.figure(figsize=(12, 6))
    plt.plot(indices, projections, 'b-', label='Projection Value')
    plt.fill_between(indices, projections, color='blue', alpha=0.1)

    plt.title("Mnemonic Projection Field Magnitude")
    plt.xlabel("Mnemonic Index (N)")
    plt.ylabel("Projection Value (sum(expanded))")
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend()

    plt.tight_layout()
    plt.savefig("mnemonic_analysis.png")
    print("Analysis complete. Visualization saved to mnemonic_analysis.png")

if __name__ == "__main__":
    run_analysis()
