#!/usr/bin/env python3
"""
merge_module_analysis.py - Visualization for MergeModuleTemporal
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from merge_module_temporal import MergeModuleTemporal
from antinomy_resolver import NeuralAntinomy

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

    # 2. Run Standard Merge Module
    module = MergeModuleTemporal(similarity_threshold=0.7, retro_convergence=0.5)
    merged_tokens, merged_embs = module(tokens, embeddings)

    # 3. Run Antinomy-Aware Merge Module
    # We'll use a very low conflict threshold to force some inconsistencies for demonstration
    resolver = NeuralAntinomy(embedding_dim=4, max_steps=5, conflict_threshold=0.01)
    module_antinomy = MergeModuleTemporal(similarity_threshold=0.7, retro_convergence=0.5, antinomy_resolver=resolver)
    merged_tokens_anti, _ = module_antinomy(tokens, embeddings)

    print(f"Original tokens: {tokens}")
    print(f"Standard Merged tokens: {merged_tokens}")
    print(f"Antinomy-Aware Merged tokens: {merged_tokens_anti}")

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Heatmap of initial similarities
    sns.heatmap(sim_matrix, annot=True, xticklabels=tokens, yticklabels=tokens,
                cmap="YlGnBu", ax=ax1)
    ax1.set_title("Initial Token Similarity Matrix")

    # Bar chart of output token counts
    labels = ["Original", "Std Merged", "Anti-Aware"]
    counts = [len(tokens), len(merged_tokens), len(merged_tokens_anti)]
    ax2.bar(labels, counts, color=['blue', 'orange', 'green'])
    ax2.set_title("Token Compression Comparison")
    ax2.set_ylabel("Number of Tokens")

    plt.tight_layout()
    plt.savefig("merge_module_analysis.png")
    print("Analysis complete. Visualization saved to merge_module_analysis.png")

if __name__ == "__main__":
    run_analysis()
