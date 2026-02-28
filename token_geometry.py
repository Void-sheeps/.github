import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse

class TokenGeometry(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # projeção linear (como uma mudança de base)
        self.linear = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Estados Pré-setados (Contextos conhecidos)
        self.preset_contexts = {}

    def register_preset(self, label, token_ids):
        """Armazena a assinatura contextual (pesos de atenção) de uma sequência conhecida."""
        with torch.no_grad():
            _, weights = self.contextual_relation(token_ids)
            self.preset_contexts[label] = weights

    def compare_context_to_preset(self, current_token_ids, label):
        """Mede a divergência entre o contexto atual e um pré-setado."""
        if label not in self.preset_contexts:
            return None

        _, current_weights = self.contextual_relation(current_token_ids)
        preset_weights = self.preset_contexts[label]

        # Diferença de Frobenius entre as matrizes de atenção
        diff = torch.norm(current_weights - preset_weights)
        return diff

    def angle_between(self, token_a, token_b):
        e1 = self.embedding(token_a)
        e2 = self.embedding(token_b)

        cos_sim = F.cosine_similarity(e1, e2)
        angle = torch.acos(torch.clamp(cos_sim, -1+1e-7, 1-1e-7))
        return angle * 180 / math.pi

    def projection(self, token_a, token_b):
        e1 = self.embedding(token_a)
        e2 = self.embedding(token_b)

        proj = (torch.dot(e1.squeeze(), e2.squeeze()) /
                torch.dot(e2.squeeze(), e2.squeeze())) * e2
        return proj

    def contextual_relation(self, token_ids):
        embeddings = self.embedding(token_ids)  # (N, d)

        # produto interno como medida de proximidade
        scores = torch.matmul(embeddings, embeddings.T)

        weights = F.softmax(scores, dim=-1)

        context = torch.matmul(weights, embeddings)
        return context, weights

def run_simulation():
    vocab_size = 10
    embedding_dim = 4
    model = TokenGeometry(vocab_size, embedding_dim)

    # Registro de Contexto Pré-setado
    hello_world = torch.tensor([1, 2]) # [HELLO, WORLD]
    model.register_preset("HELLO_WORLD", hello_world)

    token_a = torch.tensor([1])
    token_b = torch.tensor([2])

    angle = model.angle_between(token_a, token_b)
    print(f"Angle between tokens 1 and 2: {angle.item():.2f} degrees")

    proj = model.projection(token_a, token_b)
    print("Geometric projection of token 1 onto 2:", proj.detach())

    sequence = torch.tensor([1, 2, 3, 4])
    context, weights = model.contextual_relation(sequence)

    print("\nAttention Matrix (weights):")
    print(weights.detach())
    print("\nContextualized Vectors:")
    print(context.detach())

    print("\n=== Context Comparison (Raw vs Preset) ===")
    raw_seq = torch.tensor([5, 6])
    diff = model.compare_context_to_preset(raw_seq, "HELLO_WORLD")
    print(f"Divergence (Raw [5,6] vs Preset 'HELLO_WORLD'): {diff.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token Geometry Analysis")
    parser.add_argument("--simulate", action="store_true", help="Run simulation")
    args = parser.parse_args()

    if args.simulate or not any(vars(args).values()):
        run_simulation()
