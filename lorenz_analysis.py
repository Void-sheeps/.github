import torch
import matplotlib.pyplot as plt
import numpy as np
from cognitive_lorenz import CognitiveLorenzField

def run_analysis():
    print("Running Cognitive Lorenz Field Analysis...")
    dim = 4
    vL0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    vJ0 = torch.tensor([0.0, 1.0, 0.0, 0.0])

    model = CognitiveLorenzField(dim=dim)
    output = model(vL0, vJ0, steps=1000)

    # Convert to numpy for plotting
    traj_L = output["vL_traj"].detach().numpy()
    traj_J = output["vJ_traj"].detach().numpy()
    traj_z = output["lorenz_z"].detach().numpy()

    # Calculate distances over time
    distances = torch.norm(output["vL_traj"] - output["vJ_traj"], dim=1).detach().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Embedding Distance over Time (modulated by Lorenz)
    ax1.plot(distances, label="||vL - vJ||", color="#2980b9", linewidth=2)
    ax1.set_title("Embedding Convergence (Lorenz Modulated)")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Distance")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # 2. Lorenz 'z' trajectory
    ax2.plot(traj_z, label="Lorenz Z (Coupling Strength)", color="#e67e22", linewidth=2)
    ax2.set_title("Lorenz Attractor Dynamics (Z component)")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Z Value")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("lorenz_field_analysis.png")
    print("Analysis complete. Visualization saved to lorenz_field_analysis.png")

if __name__ == "__main__":
    run_analysis()
