import torch
import matplotlib.pyplot as plt
import seaborn as sns
from polynomial_attention import run_simulation

def run_analysis():
    print("Running Polynomial Structural Analysis...")

    token_list, cohesion, weights = run_simulation()

    # Create visualization side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Structural Cohesion Matrix (Inicial)
    sns.heatmap(
        cohesion.detach().numpy(),
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=token_list,
        yticklabels=token_list,
        cbar_kws={'label': 'Cohesion'},
        ax=axes[0]
    )
    axes[0].set_title("Structural Cohesion (Inicial)")
    axes[0].set_xlabel("Tokens de origem")
    axes[0].set_ylabel("Tokens destino")

    # 2. Attention Weights (Contextualizado)
    sns.heatmap(
        weights.detach().numpy(),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=token_list,
        yticklabels=token_list,
        cbar_kws={'label': 'Attention Weight'},
        ax=axes[1]
    )
    axes[1].set_title("Attention Weights (Contextualizado)")
    axes[1].set_xlabel("Tokens de origem")
    axes[1].set_ylabel("Tokens destino")

    plt.tight_layout()
    plt.savefig("polynomial_analysis.png")
    print("Analysis complete. Visualization saved to polynomial_analysis.png")

if __name__ == "__main__":
    run_analysis()
