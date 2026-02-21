#!/usr/bin/env python3
"""
Reflective Watchdog Analysis
============================

Tracks coherence metrics and system adaptation over time.
Generates visualizations for variance, entropy, and cosine similarity.
"""

import torch
import matplotlib.pyplot as plt
from reflective_watchdog import ReflectiveWatchdogModel

def run_analysis():
    print("Running Reflective Watchdog Analysis...")
    torch.manual_seed(42)
    vocab_size = 100
    d_model = 32
    seq_len = 12
    num_steps = 50

    model = ReflectiveWatchdogModel(vocab_size, d_model)
    tokens = torch.randint(0, vocab_size, (1, seq_len))

    history = {
        "gate": [],
        "variance": [],
        "entropy": [],
        "similarity": []
    }

    for _ in range(num_steps):
        output = model(tokens)
        history["gate"].append(output["gate"].item())
        history["variance"].append(output["metrics"]["variance"].item())
        history["entropy"].append(output["metrics"]["entropy"].item())
        history["similarity"].append(output["metrics"]["similarity"].item())

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Reflective Watchdog System Dynamics", fontsize=16)

    steps = range(num_steps)

    axes[0, 0].plot(steps, history["gate"], color='blue')
    axes[0, 0].set_title("Meta-Gate Activation")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Gate Value")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, history["variance"], color='green')
    axes[0, 1].set_title("Hidden State Variance")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Variance")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, history["entropy"], color='red')
    axes[1, 0].set_title("Spectral Entropy")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Entropy")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, history["similarity"], color='purple')
    axes[1, 1].set_title("Mean Cosine Similarity")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Similarity")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("watchdog_analysis.png")
    print("Analysis complete. Visualization saved to watchdog_analysis.png")

if __name__ == "__main__":
    run_analysis()
