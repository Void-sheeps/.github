#!/usr/bin/env python3
"""
merge_module_analysis.py - Visualization for MergeModuleTemporal
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from merge_module_temporal import MergeModuleTemporal

def run_analysis():
    print("Running Merge Module Temporal Analysis...")

    # Setup
    tokens = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    # Create some structured embeddings
    # alpha & beta are similar, gamma & delta are similar, epsilon & zeta are similar
    embeddings = torch.tensor([
        [1.0, 0.1, 0.0, 0.0], # alpha
        [0.9, 0.2, 0.0, 0.0], # beta
        [0.0, 0.8, 0.2, 0.0], # gamma
        [0.0, 0.7, 0.3, 0.0], # delta
        [0.0, 0.0, 0.9, 0.1], # epsilon
        [0.0, 0.0, 0.8, 0.2]  # zeta
    ])

    # 1. Calculate Initial Similarity Matrix
    n = len(tokens)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = MergeModuleTemporal.similarity_metric(embeddings[i], embeddings[j])

    # 2. Run Merge Module
    module = MergeModuleTemporal(similarity_threshold=0.2, retro_convergence=0.5)
    merged_tokens, merged_embs = module(tokens, embeddings)

    print(f"Original tokens: {tokens}")
    print(f"Merged tokens: {merged_tokens}")

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Heatmap of initial similarities
    sns.heatmap(sim_matrix, annot=True, xticklabels=tokens, yticklabels=tokens,
                cmap="YlGnBu", ax=ax1)
    ax1.set_title("Initial Token Similarity Matrix")

    # Bar chart of output token counts (simulating multiple runs or just showing the reduction)
    ax2.bar(["Original", "Merged"], [len(tokens), len(merged_tokens)], color=['blue', 'orange'])
    ax2.set_title("Token Compression via Merging")
    ax2.set_ylabel("Number of Tokens")

    plt.tight_layout()
    plt.savefig("merge_module_analysis.png")
    print("Analysis complete. Visualization saved to merge_module_analysis.png")

if __name__ == "__main__":
    run_analysis()
