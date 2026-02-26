import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# --- Camada de atenção simplificada ---
class SimpleAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, x, return_weights=False):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        # atenção escalar
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        if return_weights:
            return out, weights
        return out

def run_example_faraday():
    print("--- Exemplo 1: Faraday & Mols ---")
    vocab = ["1mol", "e", "≈", "Faraday", "example", "."]
    token_to_idx = {tok: i for i, tok in enumerate(vocab)}

    embedding_dim = 8
    sequence = ["1mol", "e", "≈", "Faraday", "example", "."]

    embeddings = nn.Embedding(len(vocab), embedding_dim)
    attention = SimpleAttention(embedding_dim)

    input_indices = torch.tensor([token_to_idx[tok] for tok in sequence])
    x = embeddings(input_indices)

    output = attention(x)

    print("Tokens originais:", sequence)
    print("\nToken embeddings originais (shape):", x.shape)
    print("\nToken embeddings após atenção (interferência contextual):")
    print(output)

def run_example_atoms():
    print("\n--- Exemplo 2: Átomos & Isótopos ---")
    vocab = [
        "H", "H-2", "H-3",  # Hidrogênio e isótopos
        "O", "O-17", "O-18", # Oxigênio e isótopos
        "C", "C-13", "C-14", # Carbono e isótopos
        "example"  # token meta-contextual (fixed case)
    ]
    token_to_idx = {tok: i for i, tok in enumerate(vocab)}

    embedding_dim = 8
    sequence = ["H", "H", "O", "example"]

    embeddings = nn.Embedding(len(vocab), embedding_dim)
    attention = SimpleAttention(embedding_dim)

    input_indices = torch.tensor([token_to_idx[tok] for tok in sequence])
    x = embeddings(input_indices)

    output = attention(x)

    print("Tokens originais:", sequence)
    print("\nEmbeddings iniciais (shape):", x.shape)
    print("\nEmbeddings após atenção (interferência contextual):")
    print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atomic Attention Simulation")
    parser.add_argument("--simulate", action="store_true", help="Executa a simulação")
    args = parser.parse_args()

    if args.simulate or not any(vars(args).values()):
        run_example_faraday()
        run_example_atoms()
