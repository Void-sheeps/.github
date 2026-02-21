#!/usr/bin/env python3
"""
Empire Silicium Framework Analysis
----------------------------------
Visualizes the reduction of harmonic entropy during the iterative deliberation process.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from empire_silicium_framework import EmpireSiliciumDeliberation

def run_analysis():
    print("Running Empire Silicium Analysis...")

    # Setup parameters
    d_model = 8
    max_steps = 15
    deliberator = EmpireSiliciumDeliberation(d_model=d_model, max_steps_N=max_steps)

    # Create a random initial token concept (high entropy)
    token_concept = torch.randn(1, 1, d_model)

    entropies = []
    state = token_concept

    # Record initial entropy
    initial_entropy = deliberator.harmonic_entropy(state).item()
    entropies.append(initial_entropy)

    # Perform deliberation steps manually to record entropy at each step
    print(f"Initial Entropy: {initial_entropy:.4f}")

    for i in range(max_steps):
        with torch.no_grad():
            # Single step of deliberation
            hypotheses, _ = deliberator.parallel_thought(state, state, state)
            state = F.relu(deliberator.hypothesis_gate(hypotheses)) + state

            # Record entropy
            entropy = deliberator.harmonic_entropy(state).item()
            entropies.append(entropy)

            if i % 3 == 0 or i == max_steps - 1:
                print(f"Step {i+1}: Entropy = {entropy:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(entropies)), entropies, marker='o', linestyle='-', color='#238636', linewidth=2)
    plt.fill_between(range(len(entropies)), entropies, alpha=0.1, color='#238636')

    plt.title("Empire Silicium Deliberation: Harmonic Entropy Convergence", fontsize=14)
    plt.xlabel("Deliberation Step", fontsize=12)
    plt.ylabel("Harmonic Entropy (H)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(range(len(entropies)))

    # Aesthetic adjustments (GitHub-like theme)
    ax = plt.gca()
    ax.set_facecolor('#0d1117')
    fig = plt.gcf()
    fig.patch.set_facecolor('#0d1117')
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['top'].set_color('#30363d')
    ax.spines['right'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')

    plt.tight_layout()
    plt.savefig("empire_silicium_analysis.png", dpi=150)
    print("Analysis complete. Visualization saved to empire_silicium_analysis.png")

if __name__ == "__main__":
    run_analysis()
