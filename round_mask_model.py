#!/usr/bin/env python3
"""
RoundModelWithHiddenMask
========================

This module implements a neural network model that incorporates dependency-based
masking on its hidden state for each output dimension. It is designed to model
systems where different outputs depend on specific subsets of the input features.

Model Architecture:
- Input: x (Current input), y_prev (Previous output)
- Hidden: ReLU(Linear(x + y_prev))
- Masking: Each output dimension j is computed by scaling the hidden state
  by the mean of its dependency mask before the final linear projection.
- Output: Sigmoid-activated vector of shape (Batch, output_dim).
"""

import torch
import torch.nn as nn
import argparse

class RoundModelWithHiddenMask(nn.Module):
    """
    RoundModelWithHiddenMask: A model that incorporates dependency-based masking
    on its hidden state for each output dimension.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dependency_sets=None):
        super(RoundModelWithHiddenMask, self).__init__()
        # input_dim + output_dim because we concatenate x and y_prev
        self.fc1 = nn.Linear(input_dim + output_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # dependency_sets: list of sets of input indices relevant for each output dimension
        self.dependency_sets = dependency_sets
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Pre-compute mask means if dependency sets are fixed
        if self.dependency_sets is not None:
            self.register_buffer('mask_means', torch.tensor([
                self.build_mask(j).mean() for j in range(output_dim)
            ]))
        else:
            self.register_buffer('mask_means', torch.ones(output_dim))

    def build_mask(self, y_index):
        """Characteristic function of the dependency set D(Y_j)"""
        mask = torch.zeros(self.input_dim)
        if self.dependency_sets is not None:
            for i in self.dependency_sets[y_index]:
                mask[i] = 1.0
        else:
            mask[:] = 1.0  # if no sets, everything is allowed
        return mask

    def forward(self, x, y_prev):
        """
        Forward pass with per-output dimension masking.
        Vectorized implementation for efficiency.

        x: (Batch, input_dim)
        y_prev: (Batch, output_dim)
        Returns: (Batch, output_dim)
        """
        # 1. Concatenate current input with previous output
        combined = torch.cat((x, y_prev), dim=-1)

        # 2. Extract hidden features
        h = self.relu(self.fc1(combined))

        # 3. Apply masking logic and final projection
        # Original logic: y_j = Sigmoid( fc2( h * mask_mean_j )_j )
        # Optimized: fc2(h * m)_j = (W(h*m) + b)_j = m*(Wh)_j + b_j

        # Linear part without bias: (Batch, output_dim)
        v_no_bias = torch.matmul(h, self.fc2.weight.t())

        # Scale each dimension by its mask mean
        v_scaled = v_no_bias * self.mask_means

        # Add bias and apply activation
        return torch.sigmoid(v_scaled + self.fc2.bias)

def run_simulation():
    print("--- RoundModelWithHiddenMask Simulation (Optimized) ---")

    # Parameters
    input_dim = 4
    output_dim = 3
    hidden_dim = 8

    # Dependency sets: Y1 depends on X1; Y2 on X1,X2; Y3 on X3,X4
    dependency_sets = [
        {0},        # Y1 depends on X1
        {0, 1},     # Y2 depends on X1 and X2
        {2, 3}      # Y3 depends on X3 e X4
    ]

    # Initialize model
    model = RoundModelWithHiddenMask(input_dim, hidden_dim, output_dim, dependency_sets)

    # Example input
    Xn = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    Y_prev = torch.zeros((1, output_dim))

    # Run model
    with torch.no_grad():
        Yn = model(Xn, Y_prev)

    print(f"Input Xn: {Xn}")
    print(f"Previous Y: {Y_prev}")
    print(f"Output Yn shape: {Yn.shape}")
    print(f"Output Yn (Batch 0): {Yn[0]}")

    # Simple check for variations based on masking
    # For j=1,2, mask mean is 0.5.
    # But note: Yn[0, 1] and Yn[0, 2] will NOT be identical now because
    # they use different weights from fc2 (the 1st and 2nd rows of W).
    # Only if the weights were identical would they be the same.

    print(f"\nMask Means: {model.mask_means.numpy()}")
    print(f"Output values: {Yn[0].numpy()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round Model with Hidden Mask Simulation")
    parser.add_argument("--simulate", action="store_true", help="Run the model simulation")
    args = parser.parse_args()

    if args.simulate or not any(vars(args).values()):
        run_simulation()
