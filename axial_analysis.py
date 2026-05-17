#!/usr/bin/env python3
"""
Axial Transformer Analysis
==========================
Simulates the AxialTransformerSystem by coupling multiple perturbation operators
and visualizing the resulting mixture fields.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from axial_transformer import (
    AxialTransformerSystem,
    Domain,
    PerturbationOperator,
    DescriptionOperator
)

def run_simulation():
    print("--- Axial Transformer System Simulation ---")
    torch.manual_seed(42)

    d_model = 64
    batch_size = 1
    seq_len = 100

    # 1. Define Domain
    # A simple 1D signal (e.g., a sine wave)
    t = torch.linspace(0, 4 * torch.pi, seq_len)
    initial_state = torch.sin(t).unsqueeze(0)  # [1, 100]
    domain = Domain(state=initial_state)

    # 2. Define Description Operator
    def describe_state(state):
        return {
            "mean": state.mean().item(),
            "std": state.std().item(),
            "max": state.max().item()
        }

    description = DescriptionOperator(describe=describe_state)

    # 3. Initialize Axial System
    system = AxialTransformerSystem(description=description, d_model=d_model)

    # 4. Define and Register Perturbation Operators

    # Op 1: Identity
    op_identity = PerturbationOperator(apply=lambda x: x)
    emb_identity = torch.randn(d_model)
    system.register(op_identity, emb_identity)

    # Op 2: Inversion
    op_inversion = PerturbationOperator(apply=lambda x: -x)
    emb_inversion = torch.randn(d_model)
    system.register(op_inversion, emb_inversion)

    # Op 3: Frequency Doubling (via simple resize-like effect or just phase shift)
    # Let's do a simple phase shift for "Perturbation"
    op_shift = PerturbationOperator(apply=lambda x: torch.roll(x, shifts=seq_len//4, dims=-1))
    emb_shift = torch.randn(d_model)
    system.register(op_shift, emb_shift)

    # Op 4: Scaling
    op_scale = PerturbationOperator(apply=lambda x: 2.0 * x)
    emb_scale = torch.randn(d_model)
    system.register(op_scale, emb_scale)

    print(f"Registered {len(system.tokens)} operators.")

    # 5. Apply Unified Dynamics
    # We provide 'base_weights' which the Router uses along with operator embeddings
    # to decide the mixture alpha_i(x).
    # Here, base_weights can represent some external control or state-derived signal.
    base_weights = torch.linspace(0, 1, len(system.tokens)).unsqueeze(0) # [1, 4]

    output = system.apply(domain, base_weights)

    new_state = output["state"]
    alpha = output["alpha"] # [1, 4]

    print("\n--- Simulation Results ---")
    print(f"Interpretation: {output['interpretation']}")
    print(f"Alpha weights: {alpha.detach().numpy()}")
    print(f"New state description: {output['description']}")

    # 6. Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: States
    ax1.plot(t.numpy(), initial_state[0].numpy(), label="Initial State (Sine)", lw=2, alpha=0.6)
    ax1.plot(t.numpy(), new_state[0].detach().numpy(), label="Transformed State (Axial Mixture)", lw=3, color='red')
    ax1.set_title("Axial Field Transformation")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Alpha Mixture Weights
    op_names = ["Identity", "Inversion", "Shift", "Scale"]
    sns.barplot(x=op_names, y=alpha[0].detach().numpy(), ax=ax2, palette="viridis")
    ax2.set_title("Operator Mixture Coefficients (alpha)")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Weight")

    plt.tight_layout()
    plt.savefig("axial_analysis.png")
    print("\nVisualization saved to axial_analysis.png")

if __name__ == "__main__":
    run_simulation()
