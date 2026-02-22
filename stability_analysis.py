#!/usr/bin/env python3
"""
stability_analysis.py - Visualize Symbolic Stability metrics.
"""

import torch
import matplotlib.pyplot as plt
from symbolic_stability import SymbolicStabilityEngine

def run_analysis():
    dim = 64
    steps = 50
    engine = SymbolicStabilityEngine(dim, momentum=0.9)

    coherences = []
    stabilities = []
    confidences = []

    print("Running stability analysis...")

    # 1. Fase estável
    for _ in range(20):
        tokens = torch.randn(10, dim) * 0.1
        out = engine(tokens)
        coherences.append(out['coherence'].item())
        stabilities.append(out['stability'].item())
        confidences.append(out['confidence'].item())

    # 2. Introdução de Drift
    for i in range(30):
        drift = i * 0.02
        tokens = torch.randn(10, dim) * 0.1 + drift
        out = engine(tokens)
        coherences.append(out['coherence'].item())
        stabilities.append(out['stability'].item())
        confidences.append(out['confidence'].item())

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(coherences, label='Coherence', color='blue')
    plt.plot(stabilities, label='Stability', color='green')
    plt.plot(confidences, label='Confidence', color='red', linewidth=2)
    plt.axvline(x=20, color='gray', linestyle='--', label='Drift Start')
    plt.title("Symbolic Stability Evolution (Drift Detection)")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("stability_analysis.png")
    print("Analysis complete. Plot saved as stability_analysis.png")

if __name__ == "__main__":
    run_analysis()
