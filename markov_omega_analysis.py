import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import argparse
from markov_omega import MarkovOmegaOperator

def run_analysis(simulate=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    N = 128
    T = 128
    operator = MarkovOmegaOperator(N=N, T=T, device=device).to(device)

    # Signal A(v) = sin(2*pi*v)
    v_grid = torch.linspace(0, 1, N, device=device)
    A = torch.sin(2 * math.pi * v_grid)

    # Target complex value: 1.0 + 0.5j
    target = torch.complex(torch.tensor(1.0), torch.tensor(0.5)).to(device)

    optimizer = optim.Adam(operator.parameters(), lr=1e-2)

    steps = 500 if simulate else 10
    losses = []

    print("Starting Markov Omega Operator training...")
    for step in range(steps):
        optimizer.zero_grad()

        omega = operator(A)

        loss_main = torch.abs(omega - target)**2
        loss_reg = 0.05 * operator.smoothness_loss()

        loss = loss_main + loss_reg

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 50 == 0:
            # Complex number formatting
            res = omega.item()
            print(f"Step {step:03d} | Loss: {loss.item():.6f} | Ω: {res.real:.4f} + {res.imag:.4f}j")

    # Visualization
    plt.figure(figsize=(15, 5))

    # 1. Loss Curve
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title("Training Loss (Markov Estimation)")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.yscale('log')

    # 2. Kernel K(v)
    plt.subplot(1, 3, 2)
    with torch.no_grad():
        K_final = operator.normalize_kernel().cpu().numpy()
        plt.plot(v_grid.cpu().numpy(), K_final, label='Learned Kernel K(v)', color='orange')
        plt.plot(v_grid.cpu().numpy(), A.cpu().numpy(), label='Signal A(v)', linestyle='--', alpha=0.5)
    plt.title("Final Normalized Kernel vs Signal")
    plt.legend()

    # 3. MCMC Sampling Trace (Final step)
    plt.subplot(1, 3, 3)
    with torch.no_grad():
        samples = operator.markov_chain().cpu().numpy()
        plt.plot(samples, alpha=0.7)
        plt.axhline(y=0.5, color='r', linestyle=':', alpha=0.3)
    plt.title("Markov Chain Trajectory (v_t)")
    plt.xlabel("Chain Step t")
    plt.ylabel("Domain v")

    plt.tight_layout()
    plt.savefig("markov_omega_analysis.png")
    print("Analysis complete. Results saved to markov_omega_analysis.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true", default=True)
    args = parser.parse_args()

    run_analysis(args.simulate)
