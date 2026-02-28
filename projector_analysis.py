import torch
import matplotlib.pyplot as plt
import numpy as np
from lorenz_projector import run_simulation

def run_analysis():
    print("Running Lorenz Projector Analysis...")
    trajectory, model = run_simulation(steps=20000)

    trajectory_np = trajectory.detach().numpy()

    fig = plt.figure(figsize=(15, 7))

    # 1. 3D Trajectory
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(trajectory_np[:, 0], trajectory_np[:, 1], trajectory_np[:, 2], lw=0.5, color='royalblue', alpha=0.7)
    ax1.set_title("Lorenz Attractor Trajectory (3D)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # 2. Integral Value Heatmap on XY Plane
    ax2 = fig.add_subplot(1, 2, 2)

    x = np.linspace(trajectory_np[:, 0].min(), trajectory_np[:, 0].max(), 100)
    y = np.linspace(trajectory_np[:, 1].min(), trajectory_np[:, 1].max(), 100)
    X, Y = np.meshgrid(x, y)

    grid_points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        Z_vals = model(grid_points).numpy().reshape(X.shape)

    cp = ax2.contourf(X, Y, Z_vals, levels=20, cmap='inferno')
    fig.colorbar(cp, ax=ax2, label="f_theta(x, y)")
    ax2.plot(trajectory_np[:1000, 0], trajectory_np[:1000, 1], 'w-', lw=0.3, alpha=0.5) # path snippet
    ax2.set_title("Integral Kernel f_theta(x, y) over XY projection")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    plt.tight_layout()
    plt.savefig("projector_analysis.png")
    print("Analysis complete. Visualization saved to projector_analysis.png")

if __name__ == "__main__":
    run_analysis()
