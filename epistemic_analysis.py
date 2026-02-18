#!/usr/bin/env python3
"""
Epistemic Analysis
==================

Visualizes cognitive state transitions and conflict detection dynamics.
"""

import torch
import matplotlib.pyplot as plt
from cognitive_epistemic_field import CognitiveField, EpistemicGate

def run_analysis():
    print("Running Epistemic Field Analysis...")

    dim = 64
    n_nodes = 32
    field = CognitiveField(dim=dim)
    gate = EpistemicGate(base_indices=torch.arange(5))

    V = torch.randn(n_nodes, dim)

    history = []
    conflicts = []

    # 50 steps of evolution
    for i in range(50):
        if i < 25:
            # Gradually pull towards a target
            target = torch.ones(dim)
            C = target.unsqueeze(0).repeat(10, 1) + 0.5 * torch.randn(10, dim)
        else:
            # Sudden shift to opposite target
            target = -torch.ones(dim)
            C = target.unsqueeze(0).repeat(10, 1) + 0.5 * torch.randn(10, dim)

        V = field.update(V, C)

        # Track mean state magnitude and conflict status
        mean_v = torch.mean(V)
        conflict, coherence, depth = gate.evaluate(V, C)

        history.append(mean_v.item())
        conflicts.append(1.0 if conflict else 0.0)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Mean Cognitive State', color='tab:blue')
    ax1.plot(history, color='tab:blue', label='Mean V')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Conflict Detected', color='tab:red')
    ax2.fill_between(range(50), conflicts, color='tab:red', alpha=0.3, label='Conflict')
    ax2.set_ylim(-0.1, 1.1)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Cognitive State Evolution & Epistemic Gating')
    fig.tight_layout()
    plt.savefig('epistemic_field_analysis.png')
    print("Analysis complete. Visualization saved to epistemic_field_analysis.png")

if __name__ == "__main__":
    run_analysis()
