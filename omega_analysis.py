#!/usr/bin/env python3
"""
Omega Operator Analysis
=======================
Trains the LearnableOmegaOperator to match a target complex response
and visualizes the results.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import numpy as np
from learnable_omega import LearnableOmegaOperator

def run_simulation(steps=500, lr=0.01):
    print("--- Learnable Omega Operator Simulation ---")
    torch.manual_seed(42)

    N = 512
    v = torch.linspace(0, 1, N)

    # Input signal: A simple sine wave
    A = torch.sin(2 * torch.pi * v)

    # Target complex response
    target = torch.tensor(1.0 + 0.5j)

    # Model
    model = LearnableOmegaOperator(kernel_size=N)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    print(f"Training for {steps} steps...")
    for step in range(steps):
        optimizer.zero_grad()

        output = model(A, v)

        # Complex L2 loss: |output - target|^2
        loss = torch.abs(output - target) ** 2

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 100 == 0:
            print(f"Step {step:03d} | Loss: {loss.item():.6f} | Output: {output.item():.4f}")

    # Final Inspection
    final_output = model(A, v)
    print("\n=== Final Parameters ===")
    print(f"D: {torch.exp(model.log_D).item():.6f}")
    print(f"phi: {model.phi.item():.6f}")
    print(f"Kernel mean: {model.kernel.mean().item():.6f}")
    print(f"Kernel std: {model.kernel.std().item():.6f}")

    print("\n=== Final Output ===")
    print(f"Actual: {final_output.item():.4f}")
    print(f"Target: {target.item():.4f}")

    # Visualization
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Training Loss
    axs[0, 0].plot(losses)
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title("Training Loss (Log Scale)")
    axs[0, 0].set_xlabel("Step")
    axs[0, 0].set_ylabel("MSE Loss")
    axs[0, 0].grid(True, which="both", ls="-", alpha=0.5)

    # 2. Signals: Input A vs Learned Kernel
    axs[0, 1].plot(v.numpy(), A.numpy(), label="Input Signal A", alpha=0.7)
    axs[0, 1].plot(v.numpy(), model.kernel.detach().numpy(), label="Learned Kernel", alpha=0.7)
    axs[0, 1].set_title("Input Signal vs. Learned Kernel")
    axs[0, 1].set_xlabel("v")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Weighted Signal
    weighted = (A * model.kernel.detach()).numpy()
    axs[1, 0].fill_between(v.numpy(), weighted, color='purple', alpha=0.3)
    axs[1, 0].plot(v.numpy(), weighted, color='purple')
    axs[1, 0].set_title("Weighted Signal (A * Kernel)")
    axs[1, 0].set_xlabel("v")
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Complex Plane: Target vs Output
    axs[1, 1].scatter([target.real.item()], [target.imag.item()], color='red', marker='x', s=100, label='Target')
    axs[1, 1].scatter([final_output.real.item()], [final_output.imag.item()], color='blue', marker='o', s=50, label='Final Output')

    # Draw a line from origin to final output
    axs[1, 1].plot([0, final_output.real.item()], [0, final_output.imag.item()], 'b--', alpha=0.5)

    axs[1, 1].set_title("Complex Response Plane")
    axs[1, 1].set_xlabel("Re")
    axs[1, 1].set_ylabel("Im")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].axhline(y=0, color='k', alpha=0.2)
    axs[1, 1].axvline(x=0, color='k', alpha=0.2)

    plt.tight_layout()
    plt.savefig("omega_analysis.png")
    print("\nVisualization saved to omega_analysis.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omega Operator Analysis")
    parser.add_argument("--simulate", action="store_true", help="Run the omega training simulation")
    parser.add_argument("--steps", type=int, default=500, help="Number of training steps")
    args = parser.parse_args()

    # Run by default if no other actions are specified, or if --simulate is passed
    if args.simulate or True: # Currently always runs as there are no other action flags
        run_simulation(steps=args.steps)
