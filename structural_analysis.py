import torch
import matplotlib.pyplot as plt
import numpy as np
from structural_field import StructuralFieldNet

def run_analysis():
    print("Running Structural Field Analysis...")
    vocab_size = 50
    embed_dim = 32
    seq_len = 20
    model = StructuralFieldNet(vocab_size, embed_dim)
    model.eval()

    # Case 1: Identical tokens
    tokens_f = torch.randint(0, vocab_size, (1, seq_len))
    tokens_g_identical = tokens_f.clone()

    # Case 2: Similar tokens (change 2 tokens)
    tokens_g_similar = tokens_f.clone()
    tokens_g_similar[0, 5] = (tokens_g_similar[0, 5] + 1) % vocab_size
    tokens_g_similar[0, 15] = (tokens_g_similar[0, 15] + 1) % vocab_size

    # Case 3: Dissimilar tokens (completely random)
    tokens_g_dissimilar = torch.randint(0, vocab_size, (1, seq_len))

    with torch.no_grad():
        dist_id = model(tokens_f, tokens_g_identical).item()
        dist_sim = model(tokens_f, tokens_g_similar).item()
        dist_dis = model(tokens_f, tokens_g_dissimilar).item()

    labels = ['Identical', 'Similar', 'Dissimilar']
    distances = [dist_id, dist_sim, dist_dis]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, distances, color=['#2ecc71', '#f1c40f', '#e74c3c'])
    plt.title('Structural Field Distance Comparison')
    plt.ylabel('MSE Distance (Structural Signature)')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', va='bottom', ha='center')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("structural_analysis.png")
    print("Analysis complete. Visualization saved to structural_analysis.png")

if __name__ == "__main__":
    run_analysis()
