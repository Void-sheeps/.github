import torch
import matplotlib.pyplot as plt
import seaborn as sns
from polynomial_attention import run_simulation

def run_analysis():
    print("Running Polynomial Structural Analysis...")

    token_list, cohesion, weights = run_simulation()

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Structural Cohesion Matrix
    sns.heatmap(
        cohesion.detach().numpy(),
        annot=True,
        fmt=".2f",
        xticklabels=token_list,
        yticklabels=token_list,
        cmap="YlGnBu",
        ax=ax1
    )
    ax1.set_title("Structural Cohesion Matrix (Cosine Similarity)")

    # 2. Structural Attention Weights
    sns.heatmap(
        weights.detach().numpy(),
        annot=True,
        fmt=".2f",
        xticklabels=token_list,
        yticklabels=token_list,
        cmap="Reds",
        ax=ax2
    )
    ax2.set_title("Structural Attention Weights")

    plt.tight_layout()
    plt.savefig("polynomial_analysis.png")
    print("Analysis complete. Visualization saved to polynomial_analysis.png")

if __name__ == "__main__":
    run_analysis()
