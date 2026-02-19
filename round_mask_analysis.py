#!/usr/bin/env python3
"""
Round Model Mask Analysis
=========================

Generates visualizations for the RoundModelWithHiddenMask.
Visualizes how different dependency masks affect the output dimensions.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from round_mask_model import RoundModelWithHiddenMask

def run_analysis():
    print("Running Round Model Mask Analysis...")

    # Setup model
    input_dim = 20
    output_dim = 10
    hidden_dim = 64

    # Create dependency sets with increasing sizes (different mask means)
    # This will demonstrate the impact of the mask scale on the output.
    dependency_sets = [set(range(int(input_dim * (j + 1) / output_dim))) for j in range(output_dim)]

    model = RoundModelWithHiddenMask(input_dim, hidden_dim, output_dim, dependency_sets)

    # Test input: constant ones to highlight masking effects
    x = torch.ones((1, input_dim))
    y_prev = torch.zeros((1, output_dim))

    with torch.no_grad():
        yn = model(x, y_prev) # Shape (1, output_dim)

    outputs = yn[0].cpu().numpy()
    mask_means = model.mask_means.cpu().numpy()

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Output Values
    sns.barplot(x=[f"Y_{i}" for i in range(output_dim)], y=outputs, ax=ax1, palette="viridis")
    ax1.set_title("Model Output per Dimension")
    ax1.set_ylabel("Value (Sigmoid)")
    ax1.set_ylim(0, 1)

    # Plot 2: Correlation between Mask Mean and Output
    ax2.scatter(mask_means, outputs, s=100, alpha=0.7, c=outputs, cmap="viridis")
    ax2.set_title("Correlation: Mask Density vs. Output Magnitude")
    ax2.set_xlabel("Mask Mean (Dependency Density)")
    ax2.set_ylabel("Output Value")
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Add trend line if there's enough variation
    if len(np.unique(mask_means)) > 1:
        z = np.polyfit(mask_means, outputs, 1)
        p = np.poly1d(z)
        ax2.plot(mask_means, p(mask_means), "r--", alpha=0.5, label="Trend")
        ax2.legend()

    plt.tight_layout()
    plt.savefig("round_mask_analysis.png")
    print("Analysis complete. Visualization saved to round_mask_analysis.png")

if __name__ == "__main__":
    run_analysis()
