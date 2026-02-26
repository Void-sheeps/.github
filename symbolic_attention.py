import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sp
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
        # Q, K, V are linear projections of x
        # x shape: [seq_len, dim]
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        # Scalar attention: Softmax(QK^T / sqrt(d)) * V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)

        if return_weights:
            return out, weights
        return out

class SymbolicAttentionEngine:
    def __init__(self, n_value=1.0):
        # N simbólico
        self.N = sp.Symbol('N', real=True)
        self.n_value = n_value

        # Vocabulário baseado na especificação do usuário
        # Nota: Corrigimos inconsistências visuais no vocab para alinhar com a lógica de N^2
        self.vocab = [
            "[(N-1)^2], [(N-1)], [(N)]",
            "[(N-1)^2], [(2N)], [(-1)]",
            "[(N-1)^2], [(0)], [(2(N-0.5))]",
            "[(N-1)^2], [(2*(N-0.5))], [(0)]"
        ]

        # Criação de embeddings simbólicos (Matrizes sp.Matrix)
        # Cada vetor possui 4 dimensões. A soma dos primeiros 3 elementos resulta em N^2.
        self.emb_vectors = [
            sp.Matrix([(self.N-1)**2, self.N-1, self.N, 0]),
            sp.Matrix([(self.N-1)**2, 2*self.N, -1, 0]),
            sp.Matrix([(self.N-1)**2, 0, 2*(self.N-0.5), 0]),
            sp.Matrix([(self.N-1)**2, 2*(self.N-0.5), 0, 0])
        ]

        self.token_to_idx = {tok: i for i, tok in enumerate(self.vocab)}
        self.idx_to_token = {i: tok for i, tok in enumerate(self.vocab)}

        self.embedding_dim = 4
        self.embeddings = nn.Embedding(len(self.vocab), self.embedding_dim)
        self.update_embeddings(n_value)

        self.attention = SimpleAttention(self.embedding_dim)

    def update_embeddings(self, n_value):
        """Atualiza os pesos do módulo nn.Embedding com base em um novo valor de N."""
        self.n_value = n_value
        numeric_list = []
        for v in self.emb_vectors:
            # Substitui N pelo valor numérico e converte para float
            vec = [float(val.subs(self.N, n_value)) for val in v]
            numeric_list.append(vec)

        with torch.no_grad():
            self.embeddings.weight.copy_(torch.tensor(numeric_list, dtype=torch.float32))

    def process_sequence(self, sequence_tokens, return_weights=False):
        """Converte tokens em índices, busca embeddings e aplica atenção."""
        input_indices = torch.tensor([self.token_to_idx[tok] for tok in sequence_tokens])
        x = self.embeddings(input_indices)
        return self.attention(x, return_weights=return_weights), x

def run_simulation(n_val=1.0):
    print(f"--- Symbolic N Attention Simulation (N={n_val}) ---")
    engine = SymbolicAttentionEngine(n_value=n_val)

    print("\nVocabulário e Embeddings Numéricos:")
    for i, tok in enumerate(engine.vocab):
        vec = engine.embeddings.weight[i].tolist()
        soma_triade = sum(vec[:3])
        print(f"Token {i}: {tok}")
        print(f"  Vetor: {[round(v, 4) for v in vec]}")
        print(f"  Soma (Triade): {soma_triade:.4f} (N^2 Alvo: {n_val**2:.4f})")

    sequence = engine.vocab
    output, x = engine.process_sequence(sequence)

    print("\nTokens na Sequência:")
    print(sequence)

    print("\nEmbeddings iniciais (x):")
    print(x)

    print("\nEmbeddings após atenção (interferência contextual):")
    print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Symbolic N Attention System")
    parser.add_argument("--simulate", action="store_true", help="Executa a simulação")
    parser.add_argument("--n_value", type=float, default=1.618, help="Valor de N para a simulação")
    args = parser.parse_args()

    if args.simulate or not any(vars(args).values()):
        run_simulation(args.n_value)
