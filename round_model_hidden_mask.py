#!/usr/bin/env python3
"""
Round Model with Hidden Mask
============================

Implements a recurrent-style model with input-to-output dependency masking.
Each output component Y_j is computed using a masked version of the hidden state.

Note: As provided, the model returns a tensor of shape [batch, output_dim, output_dim]
where each slice j contains the full output vector computed with the j-th mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class RoundModelWithHiddenMask(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dependency_sets=None):
        super(RoundModelWithHiddenMask, self).__init__()
        self.fc1 = nn.Linear(input_dim + output_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # dependency_sets: lista de conjuntos de índices de X relevantes para cada Y
        self.dependency_sets = dependency_sets
        self.input_dim = input_dim
        self.output_dim = output_dim

    def build_mask(self, y_index):
        """Função característica do conjunto D(Y_j)"""
        mask = torch.zeros(self.input_dim)
        if self.dependency_sets is not None:
            for i in self.dependency_sets[y_index]:
                mask[i] = 1.0
        else:
            mask[:] = 1.0  # se não houver conjuntos, tudo é permitido
        return mask

    def forward(self, x, y_prev):
        # concatena input atual com output anterior
        combined = torch.cat((x, y_prev), dim=-1)
        h = self.relu(self.fc1(combined))

        outputs = []
        for j in range(self.output_dim):
            if self.dependency_sets is not None:
                # constrói máscara baseada no conjunto de dependência
                mask = self.build_mask(j)
                # expande para hidden_dim (broadcast simples)
                hidden_mask = mask.mean() * torch.ones_like(h)
                masked_h = h * hidden_mask
            else:
                masked_h = h

            yj = torch.sigmoid(self.fc2(masked_h))
            outputs.append(yj)

        return torch.stack(outputs, dim=1)

def simulate_round_model():
    print("--- Round Model with Hidden Mask Simulation ---")

    input_dim = 4
    output_dim = 3
    hidden_dim = 8

    dependency_sets = [
        {0},        # Y1 depende de X1
        {0, 1},     # Y2 depende de X1 e X2
        {2, 3}      # Y3 depende de X3 e X4
    ]

    model = RoundModelWithHiddenMask(input_dim, hidden_dim, output_dim, dependency_sets)

    Xn = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    Y_prev = torch.zeros((1, output_dim))

    Yn = model(Xn, Y_prev)
    print("Input Xn:", Xn.numpy())
    print("Output Y^n shape:", Yn.shape)
    print("Output Y^n (first batch element):")
    print(Yn[0].detach().numpy())

    print("\nSimulation Complete.")

if __name__ == "__main__":
    if "--simulate" in sys.argv or len(sys.argv) == 1:
        simulate_round_model()
