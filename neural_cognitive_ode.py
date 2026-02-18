#!/usr/bin/env python3
"""
Neural Cognitive ODE System
===========================

Implements a vector field and ODE-based evolution for cognitive states.
Uses torchdiffeq for integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import sys

# ----------------------------
# Campo Vetorial Neural (Neural Vector Field)
# ----------------------------

class CognitiveVectorField(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim)
        )

    def forward(self, t, state):
        """
        state = (V, M, S)
        """
        V, M, S = state

        # concatenação estrutural (structural concatenation)
        x = torch.cat([V, M, S], dim=1)

        dV = self.net(x)

        # normalização suave (mantém na hiperesfera)
        # smooth normalization (keeps on hypersphere)
        dV = dV - (V * (V * dV).sum(dim=1, keepdim=True))

        return (dV, torch.zeros_like(M), torch.zeros_like(S))


# ----------------------------
# Sistema Neural ODE (Neural ODE System)
# ----------------------------

class NeuralCognitiveODE(nn.Module):
    def __init__(self, num_words=64, dim=128):
        super().__init__()

        self.num_words = num_words
        self.dim = dim

        self.field = CognitiveVectorField(dim)

    def forward(self, V0, M, S, t):
        state0 = (V0, M, S)
        solution = odeint(self.field, state0, t, method='dopri5')
        Vt = solution[0]  # apenas V evolui (only V evolves)
        return Vt

def simulate_neural_ode():
    print("--- Neural Cognitive ODE Simulation ---")

    num_words = 32
    dim = 64
    model = NeuralCognitiveODE(num_words=num_words, dim=dim)

    # Initial states
    V0 = torch.randn(num_words, dim)
    V0 = F.normalize(V0, dim=1)

    M = torch.randn(num_words, dim)
    S = torch.randn(num_words, dim)

    # Time points
    t = torch.linspace(0, 1, 10)

    print(f"Evolving {num_words} words in {dim}-dim space...")
    Vt = model(V0, M, S, t)

    print(f"Final state V(t) shape: {Vt.shape}")

    # Verify hypersphere constraint
    magnitudes = torch.norm(Vt[-1], dim=1)
    print(f"Mean magnitude at t=1: {torch.mean(magnitudes).item():.4f}")

    print("\nSimulation Complete.")

if __name__ == "__main__":
    if "--simulate" in sys.argv or len(sys.argv) == 1:
        simulate_neural_ode()
