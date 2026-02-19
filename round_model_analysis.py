#!/usr/bin/env python3
"""
Round Model Analysis
====================

Analyzes the dependency sets and gradient flow in the RoundModelWithHiddenMask.
Demonstrates that the current implementation uses global scaling rather than
feature isolation, and proposes a 'Strong Mask' alternative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from round_model_hidden_mask import RoundModelWithHiddenMask

def analyze_gradients():
    print("Running Round Model Gradient Analysis...")

    input_dim = 4
    output_dim = 3
    hidden_dim = 16
    dependency_sets = [{0}, {0, 1}, {2, 3}]

    model = RoundModelWithHiddenMask(input_dim, hidden_dim, output_dim, dependency_sets)

    X = torch.randn(1, input_dim, requires_grad=True)
    Y_prev = torch.zeros(1, output_dim)

    Yn = model(X, Y_prev) # [1, 3, 3]

    grad_matrix = torch.zeros(output_dim, input_dim)

    for j in range(output_dim):
        model.zero_grad()
        if X.grad is not None:
            X.grad.zero_()

        # We take the j-th component from the j-th slice (intended behavior)
        Yj = Yn[0, j, j]
        Yj.backward(retain_graph=True)
        grad_matrix[j] = X.grad.abs().squeeze()

    print("\nGradient Dependency Matrix (Output Y_j rows vs Input X_i columns):")
    print(grad_matrix.detach().numpy())

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(grad_matrix.detach().numpy(), cmap='hot', interpolation='nearest')
    plt.colorbar(label='Gradient Magnitude')
    plt.xticks(range(input_dim), [f'X{i}' for i in range(input_dim)])
    plt.yticks(range(output_dim), [f'Y{j}' for j in range(output_dim)])
    plt.title('Gradient Dependency Heatmap (Current Implementation)')
    plt.savefig('round_model_gradients.png')

    print("\nAnalysis: All outputs depend on all inputs because masking is applied AFTER the first linear layer.")
    print("To isolate dependencies, the mask should be applied to X before fc1.")
    print("Visualization saved to round_model_gradients.png")

if __name__ == "__main__":
    analyze_gradients()
