import torch
import matplotlib.pyplot as plt
import seaborn as sns
from token_geometry import TokenGeometry

def run_analysis():
    print("Running Token Geometry Analysis...")
    vocab_size = 20
    embedding_dim = 8
    model = TokenGeometry(vocab_size, embedding_dim)
    model.eval()

    # Create a sequence of tokens
    sequence = torch.arange(vocab_size)
    with torch.no_grad():
        context, weights = model.contextual_relation(sequence)

    plt.figure(figsize=(10, 8))
    sns.heatmap(weights.numpy(), annot=False, cmap="viridis")
    plt.title("Token Geometry: Contextual Relation (Attention Weights)")
    plt.xlabel("Token ID")
    plt.ylabel("Token ID")

    plt.savefig("geometry_analysis.png")
    print("Analysis complete. Visualization saved to geometry_analysis.png")

if __name__ == "__main__":
    run_analysis()
