#!/usr/bin/env python3
"""
Neural ODE Analysis
===================

Visualizes the trajectory of cognitive states evolved via Neural ODE.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from neural_cognitive_ode import NeuralCognitiveODE

def run_analysis():
    print("Running Neural ODE Analysis...")

    num_words = 16
    dim = 64
    model = NeuralCognitiveODE(num_words=num_words, dim=dim)

    V0 = torch.randn(num_words, dim)
    V0 = F.normalize(V0, dim=1)

    M = torch.randn(num_words, dim)
    S = torch.randn(num_words, dim)

    t = torch.linspace(0, 2, 50)

    with torch.no_grad():
        Vt = model(V0, M, S, t)

    # Track distance from initial state
    distances = torch.norm(Vt - V0, dim=2) # [T, N]

    # Track magnitude stability
    magnitudes = torch.norm(Vt, dim=2) # [T, N]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i in range(num_words):
        plt.plot(t.numpy(), distances[:, i].numpy(), alpha=0.5)
    plt.xlabel('Time (t)')
    plt.ylabel('Distance from V(0)')
    plt.title('Cognitive State Trajectories (Relative)')

    plt.subplot(1, 2, 2)
    plt.plot(t.numpy(), magnitudes.mean(dim=1).numpy(), 'r-', linewidth=2)
    plt.xlabel('Time (t)')
    plt.ylabel('Mean Magnitude')
    plt.ylim(0.9, 1.1)
    plt.title('Hypersphere Constraint Stability')

    plt.tight_layout()
    plt.savefig('neural_ode_analysis.png')
    print("Analysis complete. Visualization saved to neural_ode_analysis.png")

if __name__ == "__main__":
    run_analysis()
