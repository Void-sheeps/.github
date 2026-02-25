import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class DiscreteDifferentialOperator(nn.Module):
    """
    Operador Δ discreto ao longo da dimensão da sequência.
    """
    def forward(self, x):
        # x: (batch, seq_len, dim)
        return x[:, 1:, :] - x[:, :-1, :]


class StructuralSignature(nn.Module):
    """
    Extrai assinatura estrutural:
    [média, energia L2, média do Δ, energia do Δ]
    """
    def __init__(self):
        super().__init__()
        self.delta = DiscreteDifferentialOperator()

    def forward(self, x):
        delta_x = self.delta(x)

        mean = x.mean(dim=1)
        energy = (x ** 2).mean(dim=1)

        delta_mean = delta_x.mean(dim=1)
        delta_energy = (delta_x ** 2).mean(dim=1)

        return torch.cat([mean, energy, delta_mean, delta_energy], dim=-1)


class StructuralFieldNet(nn.Module):
    """
    Campo estrutural comparador de f(x) e g(x).
    """

    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        # Estado 0 explícito (vetor nulo)
        self.register_buffer("state_zero", torch.zeros(embed_dim))

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.signature = StructuralSignature()

    def forward(self, tokens_f, tokens_g):
        """
        tokens_f, tokens_g: (batch, seq_len)
        """

        # Embedding
        emb_f = self.embedding(tokens_f)
        emb_g = self.embedding(tokens_g)

        # Centralização (estado 0 como referência)
        emb_f = emb_f - self.state_zero
        emb_g = emb_g - self.state_zero

        # Assinatura estrutural
        sig_f = self.signature(emb_f)
        sig_g = self.signature(emb_g)

        # Métrica de equivalência
        distance = F.mse_loss(sig_f, sig_g, reduction="none").mean(dim=-1)

        return distance

if __name__ == "__main__":
    if "--simulate" in sys.argv:
        print("Starting StructuralFieldNet simulation...")
        vocab_size = 100
        embed_dim = 64
        seq_len = 10
        batch_size = 4

        model = StructuralFieldNet(vocab_size, embed_dim)

        tokens_f = torch.randint(0, vocab_size, (batch_size, seq_len))
        tokens_g = torch.randint(0, vocab_size, (batch_size, seq_len))

        dist = model(tokens_f, tokens_g)
        print(f"Structural Distance: {dist}")
        print("Simulation complete.")
