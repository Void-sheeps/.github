"""
NeuralDOS Analysis
==================
Simulates NeuralDOS execution and visualizes state convergence.
"""

import torch
import matplotlib.pyplot as plt
import argparse
from neural_dos import NeuralDOS

def run_simulation(steps=10):
    print("--- NeuralDOS Simulation ---")
    torch.manual_seed(42)

    vocab_size = 100
    dim = 128
    seq_len = 16
    batch_size = 4

    # Use CPU for simulation by default
    device = torch.device("cpu")

    model = NeuralDOS(vocab_size=vocab_size, dim=dim, seq_len=seq_len, T=steps, tol=1e-6).to(device)

    # Input sequence
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    # Sample program
    program = [
        0x01,  # relation (AX-modulated attention)
        0x02,  # succession (RNN)
        0x03,  # load AX (global constraint)
        0x04   # feedback (cross-attention)
    ]

    print(f"Running simulation for {steps} iterations...")
    logits, deltas = model(x, program)

    # Convert deltas to numpy for plotting, ensure it's on CPU
    deltas_np = deltas.cpu().detach().numpy()

    print("\n=== Execution Results ===")
    print(f"Logits shape: {logits.shape}")
    print(f"Deltas shape: {deltas.shape}")
    print(f"Final mean delta: {deltas_np[-1].mean():.6e}")

    # Visualization
    plt.figure(figsize=(10, 6))
    for b in range(batch_size):
        plt.plot(deltas_np[:, b], label=f'Batch {b}', marker='o')

    plt.yscale('log')
    plt.title("NeuralDOS State Convergence (Frobenius Norm Delta)")
    plt.xlabel("Iteration")
    plt.ylabel("Delta (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig("neural_dos_analysis.png")
    print("\nVisualization saved to neural_dos_analysis.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralDOS Analysis")
    parser.add_argument("--simulate", action="store_true", help="Run the NeuralDOS simulation")
    parser.add_argument("--steps", type=int, default=10, help="Number of iterations")
    args = parser.parse_args()

    if args.simulate or True:
        run_simulation(steps=args.steps)
