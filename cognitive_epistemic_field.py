#!/usr/bin/env python3
"""
Cognitive Epistemic Field & Epistemic Gate
==========================================

This module implements a dynamic cognitive state assimilation system
and an epistemic conflict detection gate.

- CognitiveField: Handles state updates (assimilation, forgetting, novelty).
- EpistemicGate: Detects logical conflicts and computes axiomatic projections.
"""

import torch
import torch.nn.functional as F
import sys

class CognitiveField:
    def __init__(self, dim=64, eta=0.1, lam=0.02, gamma=0.05):
        self.eta = eta      # aprendizado (learning)
        self.lam = lam      # esquecimento (forgetting)
        self.gamma = gamma  # ajuste de peso (weight adjustment)

    def update(self, V, C):
        """
        V: [N, d] internal state (estado interno)
        C: [M, d] current context/book (livro atual)
        """
        V_norm = F.normalize(V, dim=1)
        C_norm = F.normalize(C, dim=1)

        similarity = V_norm @ C_norm.T
        influence = similarity @ C_norm

        # assimilation + forgetting (assimilação + esquecimento)
        V_new = (1 - self.lam) * V + self.eta * influence

        # novelty calculation (cálculo de novidade)
        max_sim, _ = similarity.max(dim=1, keepdim=True)
        novelty = 1 - max_sim

        # weight adjustment (ajuste de peso)
        V_new = V_new * (1 + self.gamma * novelty)

        # global normalization (normalization global para estabilidade)
        V_new = F.normalize(V_new, dim=1)

        return V_new

class EpistemicGate:
    def __init__(self, base_indices, conflict_threshold=-0.3):
        self.base_indices = base_indices
        self.conflict_threshold = conflict_threshold

    def detect_conflict(self, V, C):
        Vn = F.normalize(V, dim=1)
        Cn = F.normalize(C, dim=1)
        alignment = torch.mean(Vn @ Cn.T)
        return alignment < self.conflict_threshold

    def axiomatic_projection(self, concept, V):
        base = V[self.base_indices]
        base = F.normalize(base, dim=1)
        concept = F.normalize(concept, dim=0)
        coherence = torch.mean(base @ concept)
        depth = 1 - coherence
        return coherence, depth

    def evaluate(self, V, C):
        if self.detect_conflict(V, C):
            central_concept = torch.mean(C, dim=0)
            coherence, depth = self.axiomatic_projection(central_concept, V)
            return True, coherence, depth
        return False, None, None

def simulate_epistemic_system():
    print("--- Cognitive Epistemic Field Simulation ---")

    dim = 64
    n_nodes = 32
    field = CognitiveField(dim=dim)
    gate = EpistemicGate(base_indices=torch.arange(5), conflict_threshold=-0.01) # Sensitive threshold

    # Initial random state
    V = torch.randn(n_nodes, dim)

    # Context 1: Harmonious
    C_harmony = V[:10] + 0.1 * torch.randn(10, dim)

    print("\nPhase 1: Harmonious Assimilation")
    V = field.update(V, C_harmony)
    conflict, coherence, depth = gate.evaluate(V, C_harmony)
    print(f"Conflict Detected: {conflict}")

    # Context 2: Conflicting
    C_conflict = -V[:10] + 0.1 * torch.randn(10, dim)

    print("\nPhase 2: Conflicting Context")
    conflict, coherence, depth = gate.evaluate(V, C_conflict)
    print(f"Conflict Detected: {conflict}")
    if conflict:
        print(f"Coherence: {coherence.item():.4f}")
        print(f"Epistemic Depth: {depth.item():.4f}")

    print("\nSimulation Complete.")

if __name__ == "__main__":
    if "--simulate" in sys.argv or len(sys.argv) == 1:
        simulate_epistemic_system()
