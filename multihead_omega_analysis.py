#!/usr/bin/env python3
"""
Multi-Head Omega Operator Analysis
==================================
Trains the MultiHeadOmegaOperator to match a vector of target complex responses.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import numpy as np
from learnable_omega import MultiHeadOmegaOperator

def run_simulation(steps=600, lr=0.01):
    print("--- Multi-Head Omega Operator Simulation ---")
    torch.manual_seed(42)

    N = 512
    H = 4  # Number of heads
    v = torch.linspace(0, 1, N)
    A = torch.sin(2 * torch.pi * v)

    # Target: Vector-valued (one per head)
    target = torch.tensor([
        1.0 + 0.5j,
        0.5 + 1.0j,
        -0.8 + 0.3j,
        0.2 - 0.9j
    ], dtype=torch.complex64)

    # Model
    model = MultiHeadOmegaOperator(num_heads=H, kernel_size=N)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    print(f"Training {H} heads for {steps} steps...")
    for step in range(steps):
        optimizer.zero_grad()

        output = model(A, v)

        # Complex vector loss: Mean squared magnitude of error
        loss = torch.mean(torch.abs(output - target) ** 2)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 100 == 0:
            print(f"Step {step:03d} | Loss: {loss.item():.6f}")

    # Final Output
    final_output = model(A, v).detach()
    print("\nFinal Output vs Target:")
    for h in range(H):
        print(f"Head {h}: Actual {final_output[h].item():.4f} | Target {target[h].item():.4f}")

    # Visualization
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Training Loss
    axs[0, 0].plot(losses)
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title("Training Loss")
    axs[0, 0].set_xlabel("Step")
    axs[0, 0].grid(True, which="both", ls="-", alpha=0.5)

    # 2. Learned Kernels
    for h in range(H):
        axs[0, 1].plot(v.numpy(), model.kernel[h].detach().numpy(), label=f"Head {h}", alpha=0.7)
    axs[0, 1].set_title("Learned Kernels")
    axs[0, 1].set_xlabel("v")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Weighted Signal (Head 0 example)
    weighted_h0 = (A * model.kernel[0].detach() * model.v_domain).numpy()
    axs[1, 0].fill_between(v.numpy(), weighted_h0, color='teal', alpha=0.3)
    axs[1, 0].plot(v.numpy(), weighted_h0, color='teal')
    axs[1, 0].set_title("Weighted Signal (Head 0)")
    axs[1, 0].set_xlabel("v")
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Complex Plane
    colors = plt.cm.rainbow(np.linspace(0, 1, H))
    for h in range(H):
        # Target
        axs[1, 1].scatter([target[h].real.item()], [target[h].imag.item()],
                          color=colors[h], marker='x', s=100, label=f'Target {h}' if h==0 else "")
        # Actual
        axs[1, 1].scatter([final_output[h].real.item()], [final_output[h].imag.item()],
                          color=colors[h], marker='o', s=50, label=f'Final {h}' if h==0 else "")
        # Line from origin
        axs[1, 1].plot([0, final_output[h].real.item()], [0, final_output[h].imag.item()],
                      color=colors[h], linestyle='--', alpha=0.3)

    axs[1, 1].set_title("Complex Output Plane (All Heads)")
    axs[1, 1].set_xlabel("Re")
    axs[1, 1].set_ylabel("Im")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].axhline(y=0, color='k', alpha=0.2)
    axs[1, 1].axvline(x=0, color='k', alpha=0.2)

    plt.tight_layout()
    plt.savefig("multihead_omega.png")
    print("\nVisualization saved to multihead_omega.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Head Omega Analysis")
    parser.add_argument("--simulate", action="store_true", help="Run multi-head simulation")
    parser.add_argument("--steps", type=int, default=600, help="Number of steps")
    args = parser.parse_args()

    # Run simulation
    run_simulation(steps=args.steps)
