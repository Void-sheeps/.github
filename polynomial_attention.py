import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sp
import argparse

# ==========================================================
# 0) Minimal mathlib Implementation
# ==========================================================

class PolynomialSpace:
    def __init__(self, variable='x', degree=2):
        self.variable = sp.Symbol(variable)
        self.degree = degree

    def canonical(self, expr):
        expanded = sp.expand(expr)

        if expanded.has(self.variable):
            poly = sp.Poly(expanded, self.variable)
            coeffs = poly.all_coeffs()
        else:
            coeffs = [expanded]

        # normalize to fixed degree
        while len(coeffs) < self.degree + 1:
            coeffs.insert(0, 0)

        coeffs = coeffs[-(self.degree + 1):]

        return torch.tensor(
            [float(c) for c in coeffs],
            dtype=torch.float32
        )

    def equivalent(self, expr_a, expr_b):
        return sp.simplify(expr_a - expr_b) == 0


# ==========================================================
# 1) Structural Attention Layer
# ==========================================================

class StructuralAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

    def forward(self, x, return_weights=False):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        scores = torch.matmul(Q, K.T) / (x.shape[-1] ** 0.5)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)

        if return_weights:
            return out, weights
        return out

def cohesion_matrix(matrix):
    normed = F.normalize(matrix, dim=1)
    return torch.matmul(normed, normed.T)

def run_simulation():
    # 1) Initialize Algebra Space
    space = PolynomialSpace(variable='x', degree=2)
    x = space.variable

    # 2) Token Registry (Algebraic Objects)
    expressions = {
        "E1": x**2,
        "E2": (x-1)**2 + (x-1) + x,              # structurally equal to x^2
        "E3": (x-2)**2 + 4*x - 4,                # structurally equal to x^2
        "E4": sp.pi**2,
        "E5": (sp.pi-1)**2 + 2*sp.pi - 1         # structurally equal to π^2
    }

    # 3) Canonical Projection (Algebra → Vector)
    vectors = {
        name: space.canonical(expr)
        for name, expr in expressions.items()
    }

    token_list = list(vectors.keys())
    embedding_matrix = torch.stack([vectors[t] for t in token_list])

    # 4) Structural Cohesion (Cosine Similarity)
    cohesion = cohesion_matrix(embedding_matrix)

    # 5) Structural Attention Layer
    attention = StructuralAttention(embedding_matrix.shape[1])
    contextualized, weights = attention(embedding_matrix, return_weights=True)

    # 6) Output
    print("=== Canonical Structural Vectors ===")
    for name in token_list:
        print(f"{name} → {vectors[name]}")

    print("\n=== Structural Cohesion Matrix ===")
    print(cohesion)

    print("\n=== Structural Equivalences (Symbolic) ===")
    for i in range(len(token_list)):
        for j in range(i+1, len(token_list)):
            if space.equivalent(
                expressions[token_list[i]],
                expressions[token_list[j]]
            ):
                print(f"{token_list[i]} ≡ {token_list[j]}")

    print("\n=== Contextualized Representations ===")
    print(contextualized)

    return token_list, cohesion, weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polynomial Structural Attention")
    parser.add_argument("--simulate", action="store_true", help="Run simulation")
    args = parser.parse_args()

    if args.simulate:
        run_simulation()
