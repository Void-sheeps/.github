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
        self.variable = sp.Symbol(variable, real=True)
        self.degree = degree

    def canonical(self, expr):
        expanded = sp.expand(expr)

        if expanded.has(self.variable):
            try:
                poly = sp.Poly(expanded, self.variable)
                coeffs = poly.all_coeffs()
            except sp.PolynomialError:
                # Handle cases where expansion results in non-polynomials (e.g. Abs(x))
                coeffs = [expanded]
        else:
            coeffs = [expanded]

        # normalize to fixed degree
        while len(coeffs) < self.degree + 1:
            coeffs.insert(0, 0)

        coeffs = coeffs[-(self.degree + 1):]

        try:
            return torch.tensor(
                [float(c) for c in coeffs],
                dtype=torch.float32
            )
        except TypeError:
            # Fallback for non-numeric coefficients (symbolic remnants)
            return torch.zeros(self.degree + 1)

    def equivalent(self, expr_a, expr_b):
        return sp.simplify(expr_a - expr_b) == 0

    def detect_non_injective_inversion(self, expr):
        issues = []

        # Check the expression itself after simplification
        simplified_top = sp.simplify(expr)
        if simplified_top.has(sp.Abs):
            issues.append(f"Implicit absolute value introduced: {simplified_top}")

        for node in sp.preorder_traversal(expr):
            # Detect sqrt(expr)
            if isinstance(node, sp.Pow) and node.exp == sp.Rational(1, 2):
                inner = node.base

                if isinstance(inner, sp.Pow) and inner.exp % 2 == 0:
                    issues.append(
                        f"Non-injective inversion: sqrt({inner})"
                    )

                # Check if simplification of this node introduces Abs
                simplified = sp.simplify(node)
                if simplified.has(sp.Abs) and f"Implicit absolute value introduced: {simplified}" not in issues:
                    issues.append(
                        f"Implicit absolute value introduced: {simplified}"
                    )

        return list(set(issues))

class DependencyAwarePolynomialSpace(PolynomialSpace):
    def __init__(self, variable='x', degree=2):
        super().__init__(variable, degree)
        self.dependency_records = {}  # name -> constraints

    def register_expression(self, name, expr):
        # 1. Compute canonical vector (as before)
        vec = self.canonical(expr)
        # 2. Analyze dependencies
        constraints = self._extract_constraints(expr)
        self.dependency_records[name] = constraints
        return vec

    def _extract_constraints(self, expr):
        # Use SymPy to traverse the expression
        constraints = {}
        for sub in sp.preorder_traversal(expr):
            if sub.is_Pow and sub.exp == sp.Rational(1, 2):
                # sqrt(arg) requires arg >= 0
                arg = sub.base
                # if arg depends on the variable, record inequality
                if self.variable in arg.free_symbols:
                    # e.g., arg >= 0
                    constraints.setdefault(self.variable, []).append(
                        sp.Ge(arg, 0)
                    )
            elif sub.func == sp.log:
                arg = sub.args[0]
                if self.variable in arg.free_symbols:
                    constraints.setdefault(self.variable, []).append(
                        sp.Gt(arg, 0)
                    )
            # ... handle Abs, trig inverses, etc.
        return constraints

    def validate_input(self, name, value):
        constraints = self.dependency_records.get(name, {})
        var = self.variable
        if var in constraints:
            for cond in constraints[var]:
                # Force boolean evaluation
                result = bool(cond.subs({var: value}))
                if not result:
                    return False
        return True


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
    space = DependencyAwarePolynomialSpace(variable='x', degree=2)
    x = space.variable

    # 2) Token Registry (Algebraic Objects)
    expressions = {
        "E1": x**2,
        "E2": (x-1)**2 + (x-1) + x,              # structurally equal to x^2
        "E3": (x-2)**2 + 4*x - 4,                # structurally equal to x^2
        "E4": sp.pi**2,
        "E5": (sp.pi-1)**2 + 2*sp.pi - 1,        # structurally equal to π^2
        "E6": sp.sqrt(x-5)                       # Requires x >= 5
    }

    # 3) Canonical Projection (Algebra → Vector)
    vectors = {}
    for name, expr in expressions.items():
        try:
            vectors[name] = space.register_expression(name, expr)
        except (TypeError, sp.PolynomialError):
            continue

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

    print("\n=== Dependency Analysis ===")
    for name, expr in expressions.items():
        issues = space.detect_non_injective_inversion(expr)
        for issue in issues:
            print(f"{name} → {issue}")

    print("\n=== Contextualized Representations ===")
    print(contextualized)

    print("\n=== Input Validation (Dependency Aware) ===")
    test_values = [0, 6]
    for val in test_values:
        is_valid = space.validate_input("E6", val)
        print(f"E6 validation for x={val}: {'VALID' if is_valid else 'INVALID'}")

    return token_list, cohesion, weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polynomial Structural Attention")
    parser.add_argument("--simulate", action="store_true", help="Run simulation")
    args = parser.parse_args()

    if args.simulate:
        run_simulation()
