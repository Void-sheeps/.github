import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from atomic_attention import SimpleAttention

def run_analysis():
    print("Running Atomic Attention Analysis...")

    # Setup
    vocab = ["H", "H-2", "H-3", "O", "O-17", "O-18", "C", "C-13", "C-14", "example"]
    token_to_idx = {tok: i for i, tok in enumerate(vocab)}
    embedding_dim = 8
    sequence = ["H", "H", "O", "C", "O-18", "example"]

    # Initialize model
    embeddings = nn.Embedding(len(vocab), embedding_dim)
    attention = SimpleAttention(embedding_dim)

    # Process
    input_indices = torch.tensor([token_to_idx[tok] for tok in sequence])
    x = embeddings(input_indices)

    # Forward pass returning weights
    output, weights = attention(x, return_weights=True)

    # Convert weights to numpy for plotting
    attn_weights = weights.detach().numpy()

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, annot=True, xticklabels=sequence, yticklabels=sequence, cmap="viridis")
    plt.title("Atomic Attention Weights Heatmap")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")

    plt.tight_layout()
    plt.savefig("atomic_attention_analysis.png")
    print("Analysis complete. Visualization saved to atomic_attention_analysis.png")

if __name__ == "__main__":
    run_analysis()
