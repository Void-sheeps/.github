#!/usr/bin/env python3
"""
IntegralFormer Analysis
=======================
Visualizes the kernel matrix and state transitions of the IntegralFormer.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from integral_former import IntegralFormer

def run_analysis(seq_len=16, d_model=32, vocab_size=20):
    print(f"--- IntegralFormer Analysis (Seq={seq_len}, Dim={d_model}) ---")
    torch.manual_seed(123)

    model = IntegralFormer(seq_len=seq_len, d_model=d_model, vocab_size=vocab_size)

    # Synthetic input sequence
    input_tokens = torch.randint(0, vocab_size, (1, seq_len))

    with torch.no_grad():
        # Get embeddings
        X = model.token_emb(input_tokens) + model.pos_emb

        # Get kernel
        K_kernel = model.get_kernel(X)

        # Get output
        Y = model(input_tokens)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Kernel Heatmap
    sns.heatmap(K_kernel[0].numpy(), ax=ax1, cmap="viridis", cbar_kws={'label': 'Kernel Strength'})
    ax1.set_title("Gaussian Kernel Matrix K(i, t)")
    ax1.set_xlabel("t (Time/Sequence Index)")
    ax1.set_ylabel("i (Query Index)")

    # 2. Embedding Magnitude Change
    x_norms = torch.norm(X[0], dim=-1).numpy()
    y_norms = torch.norm(Y[0], dim=-1).numpy()

    indices = range(seq_len)
    ax2.bar(indices, x_norms, alpha=0.5, label='Input Embeddings (X)', color='blue')
    ax2.bar(indices, y_norms - x_norms, bottom=x_norms, alpha=0.5, label='Integral Addition', color='red')
    ax2.set_title("Embedding Magnitude: Input vs. Integral Output")
    ax2.set_xlabel("Token Index")
    ax2.set_ylabel("L2 Norm")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("integral_analysis.png")
    print("Analysis complete. Visualization saved to integral_analysis.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IntegralFormer Analysis")
    parser.add_argument("--simulate", action="store_true", help="Run the analysis simulation")
    args = parser.parse_args()

    if args.simulate or True:
        run_analysis()
