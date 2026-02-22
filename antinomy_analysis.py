#!/usr/bin/env python3
"""
antinomy_analysis.py - Visualization for Neural-Symbolic Antinomy Resolver
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from antinomy_resolver import NeuralAntinomy, Atom

def run_analysis():
    print("Running Neural-Symbolic Antinomy Analysis...")

    tokens = ["Justice", "Mercy", "Law", "Chaos", "Order", "Freedom"]
    atoms = [Atom(t) for t in tokens]

    # Create specific embeddings to simulate some conflicts
    # Group 1: Justice, Law, Order
    # Group 2: Chaos, Freedom
    # Mercy is somewhat in between
    embeddings = torch.zeros(len(tokens), 16)
    embeddings[0, 0:4] = 1.0 # Justice
    embeddings[1, 2:6] = 1.0 # Mercy
    embeddings[2, 0:2] = 1.0 # Law
    embeddings[3, 8:12] = 1.0 # Chaos
    embeddings[4, 0:4] = 0.8 # Order
    embeddings[5, 10:14] = 1.0 # Freedom

    # Add some noise
    embeddings += torch.randn_like(embeddings) * 0.1

    model = NeuralAntinomy(embedding_dim=16, max_steps=20, conflict_threshold=0.3)

    with torch.no_grad():
        updated_embs, conflict_scores, inconsistent = model(embeddings, atoms)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar chart of conflict scores
    sns.barplot(x=tokens, y=conflict_scores.numpy(), palette="magma", ax=ax1)
    ax1.set_title("Conflict Scores per Token")
    ax1.set_ylabel("Conflict Score (1 - max_similarity)")
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.3, color='r', linestyle='--', label='Threshold')
    ax1.legend()

    # Heatmap of similarity matrix (after update)
    num_tokens = updated_embs.size(0)
    sim_matrix = torch.zeros(num_tokens, num_tokens)
    for i in range(num_tokens):
        for j in range(num_tokens):
            sim_matrix[i, j] = torch.nn.functional.cosine_similarity(
                updated_embs[i].unsqueeze(0), updated_embs[j].unsqueeze(0)
            )

    sns.heatmap(sim_matrix.numpy(), annot=True, xticklabels=tokens, yticklabels=tokens,
                cmap="YlGnBu", ax=ax2)
    ax2.set_title("Post-Refinement Similarity Matrix")

    plt.tight_layout()
    plt.savefig("antinomy_analysis.png")
    print(f"Analysis complete. Visualization saved to antinomy_analysis.png")
    print(f"Inconsistent Atoms detected: {inconsistent}")

if __name__ == "__main__":
    run_analysis()
