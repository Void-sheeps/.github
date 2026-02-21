from __future__ import annotations
"""
Empire Silicium Framework
-------------------------
Integrates:
1. Symbolic tokens (container Atoms)
2. Token fusion metric (prime token extraction)
3. Iterative deliberation (EmpireSiliciumDeliberation)
4. Propositional logic evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import List, Dict, Any, Optional, Set, Tuple, Union

# ==============================
# Propositional Logic Core
# ==============================

class Formula:
    """Base class for propositional logic formulas."""
    def __invert__(self) -> 'Not':
        return Not(self)

    def __and__(self, other: 'Formula') -> 'And':
        return And(self, other)

    def __or__(self, other: 'Formula') -> 'Or':
        return Or(self, other)

    def __rshift__(self, other: 'Formula') -> 'Implies':
        return Implies(self, other)

    def __lshift__(self, other: 'Formula') -> 'Iff':
        return Iff(self, other)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Formula) and self.__class__ == other.__class__ and self.eq(other)

    def v(self, valuation: Set['Atom']) -> bool:
        """Evaluate formula under a valuation (set of true atoms)."""
        raise NotImplementedError("Plain formula cannot be valuated")

    # Tableau methods (internal use)
    def _t(self, left: List['Formula'], right: List['Formula']) -> Optional[Set['Atom']]:
        """Tableau proof procedure (simplified)."""
        while True:
            found = True
            for item in left:
                if item in right:
                    return None
                if not isinstance(item, Atom):
                    left.remove(item)
                    tup = item._tleft(left, right)
                    left, right = tup[0]
                    if len(tup) > 1:
                        v = self._t(*tup[1])
                        if v is not None:
                            return v
                    found = False
                    break
            for item in right:
                if item in left:
                    return None
                if not isinstance(item, Atom):
                    right.remove(item)
                    tup = item._tright(left, right)
                    left, right = tup[0]
                    if len(tup) > 1:
                        v = self._t(*tup[1])
                        if v is not None:
                            return v
                    found = False
                    break
            if found:
                return set(left)

    def t(self) -> Optional[Set['Atom']]:
        """Run tableau on the formula."""
        return self._t([], [self])

    def _tleft(self, left: List['Formula'], right: List['Formula']) -> Tuple[Tuple[List['Formula'], List['Formula']], ...]:
        raise NotImplementedError

    def _tright(self, left: List['Formula'], right: List['Formula']) -> Tuple[Tuple[List['Formula'], List['Formula']], ...]:
        raise NotImplementedError


class BinOp(Formula):
    """Base class for binary operators."""
    def __init__(self, lchild: Formula, rchild: Formula) -> None:
        self.lchild = lchild
        self.rchild = rchild

    def __str__(self) -> str:
        return f'({self.lchild} {self.op} {self.rchild})'

    def eq(self, other: 'BinOp') -> bool:
        return self.lchild == other.lchild and self.rchild == other.rchild


class And(BinOp):
    op = '∧'
    def v(self, valuation: Set[Atom]) -> bool:
        return self.lchild.v(valuation) and self.rchild.v(valuation)

    def _tleft(self, left: List[Formula], right: List[Formula]) -> Tuple[Tuple[List[Formula], List[Formula]], ...]:
        return (left + [self.lchild, self.rchild], right),

    def _tright(self, left: List[Formula], right: List[Formula]) -> Tuple[Tuple[List[Formula], List[Formula]], ...]:
        return (left, right + [self.lchild]), (left, right + [self.rchild])


class Or(BinOp):
    op = '∨'
    def v(self, valuation: Set[Atom]) -> bool:
        return self.lchild.v(valuation) or self.rchild.v(valuation)

    def _tleft(self, left: List[Formula], right: List[Formula]) -> Tuple[Tuple[List[Formula], List[Formula]], ...]:
        return (left + [self.lchild], right), (left + [self.rchild], right)

    def _tright(self, left: List[Formula], right: List[Formula]) -> Tuple[Tuple[List[Formula], List[Formula]], ...]:
        return (left, right + [self.lchild, self.rchild]),


class Implies(BinOp):
    op = '→'
    def v(self, valuation: Set[Atom]) -> bool:
        return not self.lchild.v(valuation) or self.rchild.v(valuation)

    def _tleft(self, left: List[Formula], right: List[Formula]) -> Tuple[Tuple[List[Formula], List[Formula]], ...]:
        return (left + [self.rchild], right), (left, right + [self.lchild])

    def _tright(self, left: List[Formula], right: List[Formula]) -> Tuple[Tuple[List[Formula], List[Formula]], ...]:
        return (left + [self.lchild], right + [self.rchild]),


class Iff(BinOp):
    op = '↔'
    def v(self, valuation: Set[Atom]) -> bool:
        return self.lchild.v(valuation) is self.rchild.v(valuation)

    def _tleft(self, left: List[Formula], right: List[Formula]) -> Tuple[Tuple[List[Formula], List[Formula]], ...]:
        return (left + [self.lchild, self.rchild], right), (left, right + [self.lchild, self.rchild])

    def _tright(self, left: List[Formula], right: List[Formula]) -> Tuple[Tuple[List[Formula], List[Formula]], ...]:
        return (left + [self.lchild], right + [self.rchild]), (left + [self.rchild], right + [self.lchild])


class Not(Formula):
    def __init__(self, child: Formula) -> None:
        self.child = child

    def v(self, valuation: Set[Atom]) -> bool:
        return not self.child.v(valuation)

    def __str__(self) -> str:
        return '¬' + str(self.child)

    def eq(self, other: 'Not') -> bool:
        return self.child == other.child

    def _tleft(self, left: List[Formula], right: List[Formula]) -> Tuple[Tuple[List[Formula], List[Formula]], ...]:
        return (left, right + [self.child]),

    def _tright(self, left: List[Formula], right: List[Formula]) -> Tuple[Tuple[List[Formula], List[Formula]], ...]:
        return (left + [self.child], right),


class Atom(Formula):
    def __init__(self, name: str) -> None:
        self.name = name

    def __hash__(self) -> int:
        return hash(self.name)

    def v(self, valuation: Set['Atom']) -> bool:
        return self in valuation

    def __str__(self) -> str:
        return str(self.name)

    __repr__ = __str__

    def eq(self, other: 'Atom') -> bool:
        return self.name == other.name


# ==============================
# Token Fusion Utilities
# ==============================

def token_metric(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """Compute overlap metric between two token embeddings."""
    shared = (vec1 * vec2).sum().item()
    return shared / vec1.numel()


def check_mergeable(vec1: torch.Tensor, vec2: torch.Tensor, threshold: float = 0.5) -> bool:
    """Determine if two tokens can be merged based on metric."""
    return token_metric(vec1, vec2) >= threshold


def merge_tokens(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """Merge two token embeddings by elementwise clamping sum."""
    return torch.clamp(vec1 + vec2, max=1.0)


def get_prime_tokens(
    tokens: List[str],
    embeddings: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[List[str], List[torch.Tensor]]:
    """
    Recursively fuse tokens until no more merges occur.

    Args:
        tokens: List of token names.
        embeddings: Tensor of shape [num_tokens, dim].
        threshold: Merge threshold for token_metric.

    Returns:
        (prime_tokens, prime_embeddings) after fusion.
    """
    # Work with mutable lists
    current_tokens = tokens.copy()
    current_embs = [embeddings[i].clone() for i in range(embeddings.size(0))]

    while True:
        n = len(current_tokens)
        if n <= 1:
            break

        merged_indices = set()
        new_tokens = []
        new_embs = []
        merge_occurred = False

        # Compare all pairs
        for i in range(n):
            if i in merged_indices:
                continue
            best_j = None
            best_score = threshold  # only consider above threshold
            for j in range(i + 1, n):
                if j in merged_indices:
                    continue
                score = token_metric(current_embs[i], current_embs[j])
                if score >= best_score:
                    best_score = score
                    best_j = j

            if best_j is not None:
                # Merge i and best_j
                merged_emb = merge_tokens(current_embs[i], current_embs[best_j])
                merged_token = f"{current_tokens[i]}|{current_tokens[best_j]}"
                new_tokens.append(merged_token)
                new_embs.append(merged_emb)
                merged_indices.add(i)
                merged_indices.add(best_j)
                merge_occurred = True
            else:
                # Keep i as is
                new_tokens.append(current_tokens[i])
                new_embs.append(current_embs[i])
                merged_indices.add(i)

        # Update current lists
        current_tokens = new_tokens
        current_embs = new_embs

        if not merge_occurred:
            break

    return current_tokens, current_embs


# ==============================
# Deliberation Module
# ==============================

class EmpireSiliciumDeliberation(nn.Module):
    """
    Iterative deliberation with parallel hypothesis generation and harmonic entropy.
    """
    def __init__(self, d_model: int = 8, num_heads: int = 2, max_steps_N: int = 5):
        super().__init__()
        self.max_steps_N = max_steps_N
        self.parallel_thought = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=False  # default: (seq_len, batch, d_model)
        )
        self.hypothesis_gate = nn.Linear(d_model, d_model)

    def forward(self, token_concept: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_concept: Input tensor of shape (seq_len, batch, d_model)
                           or (batch, d_model) – will be unsqueezed if needed.

        Returns:
            Refined token concept of same shape as input.
        """
        # Ensure 3D shape (seq_len, batch, d_model)
        if token_concept.dim() == 2:
            token_concept = token_concept.unsqueeze(0)  # add seq_len=1

        state = token_concept
        for _ in range(self.max_steps_N):
            # MultiheadAttention returns (attn_output, attn_output_weights)
            hypotheses, _ = self.parallel_thought(state, state, state)
            updated = F.relu(self.hypothesis_gate(hypotheses)) + state
            entropy = self.harmonic_entropy(updated)
            state = updated
            if entropy < 0.1:
                break
        return state

    @staticmethod
    def harmonic_entropy(tensor: torch.Tensor) -> torch.Tensor:
        """Compute mean harmonic entropy over last dimension."""
        probs = F.softmax(tensor, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy.mean()


# ==============================
# Propositional Evaluator Helper
# ==============================

class PropositionalEvaluator:
    """Helper to create and evaluate formulas over stable tokens."""
    def __init__(self, stable_tokens: List[str]):
        self.atoms = {name: Atom(name) for name in stable_tokens}
        self.all_atoms_set = set(self.atoms.values())

    def evaluate(self, formula: Formula) -> bool:
        """Evaluate formula under full valuation (all atoms true)."""
        return formula.v(self.all_atoms_set)

    def create_example_formulas(self) -> Dict[str, Formula]:
        """Create some example formulas if enough tokens exist."""
        names = list(self.atoms.keys())
        formulas = {}
        if len(names) >= 2:
            formulas['f1'] = self.atoms[names[0]] | self.atoms[names[1]]
        if len(names) >= 3:
            formulas['f2'] = self.atoms[names[2]] & self.atoms[names[0]]
            formulas['f3'] = ~self.atoms[names[1]]
        return formulas


# ==============================
# Example Usage / Simulation
# ==============================

def run_simulation():
    # Example container features (8‑dim vectors)
    container_features = {
        "namedtuple":  [1, 0, 0, 0, 0, 0, 0, 0],
        "deque":       [0, 1, 0, 0, 0, 0, 0, 0],
        "ChainMap":    [0, 0, 1, 0, 0, 0, 0, 0],
        "Counter":     [0, 0, 0, 1, 0, 0, 0, 0],
        "OrderedDict": [0, 0, 0, 0, 1, 0, 0, 0],
        "defaultdict": [0, 0, 0, 0, 0, 1, 0, 0],
        "UserDict":    [0, 0, 0, 0, 0, 0, 1, 0],
        "UserList":    [0, 0, 0, 0, 0, 0, 0, 1],
        "UserString":  [1, 1, 0, 0, 0, 0, 0, 0],  # hybrid example
    }

    tokens = list(container_features.keys())
    embeddings = torch.tensor([container_features[t] for t in tokens], dtype=torch.float)

    print("\n--- 1. Prime Token Fusion ---")
    prime_tokens, prime_embs = get_prime_tokens(tokens, embeddings, threshold=0.5)
    token_vectors = torch.stack(prime_embs)  # [num_prime, d_model]
    print(f"Fused {len(tokens)} tokens into {len(prime_tokens)} prime tokens.")
    print(f"Prime tokens: {prime_tokens}")

    # 2. Deliberation on each token
    print("\n--- 2. Iterative Deliberation ---")
    deliberator = EmpireSiliciumDeliberation(d_model=embeddings.shape[1])
    final_states = []
    for vec in token_vectors:
        # Add batch dimension -> (seq_len=1, batch=1, d_model)
        vec = vec.unsqueeze(0).unsqueeze(1)  # [1, 1, d_model]
        refined = deliberator(vec)           # [1, 1, d_model]
        final_states.append(refined.squeeze())  # [d_model]

    final_states = torch.stack(final_states)  # [num_prime, d_model]

    # 3. Filter stable tokens by low entropy
    entropies = torch.tensor([
        deliberator.harmonic_entropy(state.unsqueeze(0).unsqueeze(0)).item()
        for state in final_states
    ])
    stable_tokens = [prime_tokens[i] for i, e in enumerate(entropies) if e < 0.1]
    print(f"Identified {len(stable_tokens)} stable tokens with harmonic entropy < 0.1")

    # 4. Propositional evaluation
    print("\n--- 4. Propositional Logic Evaluation ---")
    if stable_tokens:
        evaluator = PropositionalEvaluator(stable_tokens)
        formulas = evaluator.create_example_formulas()
        print("Stable tokens:", stable_tokens)
        for name, formula in formulas.items():
            print(f"{name} = {formula}  ->  {evaluator.evaluate(formula)}")
    else:
        print("No stable tokens found.")

    print("\nSimulation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Empire Silicium Framework Simulation")
    parser.add_argument("--simulate", action="store_true", help="Run the framework simulation")
    args = parser.parse_args()

    if args.simulate or not any(vars(args).values()):
        run_simulation()
