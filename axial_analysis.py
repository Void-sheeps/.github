#!/usr/bin/env python3
"""
Axial Transformer Analysis
==========================
Analyzes the AxialTransformerSystem by applying various perturbation operators
to a domain and visualizing the result.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import numpy as np
from axial_transformer import (
    Domain, PerturbationOperator, DescriptionOperator,
    AxialTransformerSystem
)

def run_simulation():
    print("--- Axial Transformer System Simulation ---")
    torch.manual_seed(42)

    d_model = 64

    # 1. Define Domain
    # A simple 2D vector field or just a latent vector
    initial_state = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    domain = Domain(state=initial_state)
    print(f"Initial State: {initial_state.tolist()}")

    # 2. Define Description Operator
    # Simple L2 norm as description
    def simple_describe(state):
        return torch.norm(state).unsqueeze(0)

    description_op = DescriptionOperator(describe=simple_describe)

    # 3. Define Perturbation Operators

    # Op A: Scaling
    def op_a_apply(state):
        return state * 1.5
    op_a = PerturbationOperator(
        apply=op_a_apply,
        embedding=torch.randn(d_model)
    )

    # Op B: Translation/Shift
    def op_b_apply(state):
        return state + 2.0
    op_b = PerturbationOperator(
        apply=op_b_apply,
        embedding=torch.randn(d_model)
    )

    # Op C: Negation
    def op_c_apply(state):
        return -state
    op_c = PerturbationOperator(
        apply=op_c_apply,
        embedding=torch.randn(d_model)
    )

    # 4. Initialize System and Register Operators
    system = AxialTransformerSystem(description=description_op, d_model=d_model)
    system.register(op_a, weight=1.0)
    system.register(op_b, weight=0.5)
    system.register(op_c, weight=0.2)

    # 5. Apply System
    result = system.apply(domain)

    print("\n=== Simulation Result ===")
    print(f"Final State: {result['state'].tolist()}")
    print(f"Description (Norm): {result['description'].item():.4f}")
    print(f"Weights: {result['weights']}")

    # 6. Visualization
    operators = ['Scale (1.5x)', 'Shift (+2.0)', 'Negate (-1x)']
    weights = result['weights']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart for weights
    bars = ax1.bar(operators, weights, color=['#3498db', '#9b59b6', '#e67e22'])
    ax1.set_title("Learned Axial Weights")
    ax1.set_ylabel("Attention Weight (Softmax)")
    ax1.set_ylim(0, 1)
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom')

    # State visualization (simple 1D comparison)
    x_indices = np.arange(len(initial_state))
    ax2.plot(x_indices, initial_state.detach().numpy(), marker='o', label='Initial State', linestyle='--')
    ax2.plot(x_indices, result['state'].detach().numpy(), marker='s', label='Final State')
    ax2.set_title("State Transformation")
    ax2.set_xlabel("Component Index")
    ax2.set_ylabel("Value")
    ax2.set_xticks(x_indices)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("axial_analysis.png")
    print("\nVisualization saved to axial_analysis.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Axial Transformer Analysis")
    parser.add_argument("--simulate", action="store_true", help="Run the axial transformer simulation")
    args = parser.parse_args()

    if args.simulate:
        run_simulation()
    else:
        print("Use --simulate to run the analysis.")
