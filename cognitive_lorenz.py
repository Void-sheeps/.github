import torch
import torch.nn as nn

class CognitiveLorenzField(nn.Module):
    def __init__(self, dim=64, alpha=1.0, beta=0.2, gamma=0.5, sigma=10.0, rho=28.0, lorenz_beta=8/3, dt=0.01, device="cpu"):
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.device = device
        self.alpha0 = alpha
        self.beta_game = beta
        self.gamma = gamma
        self.sigma = sigma
        self.rho = rho
        self.lorenz_beta = lorenz_beta

    def intersection(self, vL, vJ):
        dot = torch.dot(vL, vJ)
        norm_sq = torch.dot(vL, vL) + 1e-8
        return (dot / norm_sq) * vL

    def payoff_gradients(self, vL, vJ):
        grad_L = vL
        grad_J = -(vJ - vL)
        return grad_L, grad_J

    def forward(self, vL, vJ, steps=500):
        vL = vL.clone().to(self.device)
        vJ = vJ.clone().to(self.device)
        x = torch.tensor(1.0, device=self.device)
        y = torch.tensor(1.0, device=self.device)
        z = torch.tensor(1.0, device=self.device)
        traj_L, traj_J, traj_z = [], [], []

        for _ in range(steps):
            # Lorenz dynamics
            dx = self.sigma * (y - x)
            dy = x * (self.rho - z) - y
            dz = x * y - self.lorenz_beta * z
            x = x + self.dt * dx
            y = y + self.dt * dy
            z = z + self.dt * dz

            alpha = self.alpha0 + self.gamma * z
            I = self.intersection(vL, vJ)
            gL, gJ = self.payoff_gradients(vL, vJ)

            dvL = alpha * (I - vL) + self.beta_game * gL
            dvJ = alpha * (I - vJ) + self.beta_game * gJ
            vL = vL + self.dt * dvL
            vJ = vJ + self.dt * dvJ

            traj_L.append(vL.clone())
            traj_J.append(vJ.clone())
            traj_z.append(z.clone())

        return {"vL_traj": torch.stack(traj_L),
                "vJ_traj": torch.stack(traj_J),
                "lorenz_z": torch.stack(traj_z)}

if __name__ == "__main__":
    dim = 4
    vL0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    vJ0 = torch.tensor([0.0, 1.0, 0.0, 0.0])

    model = CognitiveLorenzField(dim=dim)
    output = model(vL0, vJ0, steps=200)

    print("\n--- Cognitive Lorenz Field Simulation ---")
    print("Initial distance:", torch.norm(vL0 - vJ0).item())
    print("Final distance between vL and vJ embeddings:", torch.norm(output["vL_traj"][-1] - output["vJ_traj"][-1]).item())
    print("Final Lorenz attractor 'z' value:", output["lorenz_z"][-1].item())
