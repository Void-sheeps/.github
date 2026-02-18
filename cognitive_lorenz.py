#!/usr/bin/env python3
"""
Cognitive Lorenz Game Field
===========================

Embedding dynamics + Concept intersection +
Game-theoretic gradient + Lorenz attractor coupling
"""

import torch
import torch.nn as nn
import argparse


class CognitiveLorenzField(nn.Module):

    def __init__(
        self,
        dim=64,
        alpha=1.0,
        beta=0.2,
        gamma=0.5,
        sigma=10.0,
        rho=28.0,
        lorenz_beta=8/3,
        dt=0.01,
        device="cpu"
    ):
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.device = device

        # Cognitive coupling
        self.alpha0 = alpha
        self.beta_game = beta
        self.gamma = gamma

        # Lorenz parameters
        self.sigma = sigma
        self.rho = rho
        self.lorenz_beta = lorenz_beta

    def intersection(self, vL, vJ):
        # Symmetric projection-based intersection
        dot = torch.dot(vL, vJ)
        norm_sq = torch.dot(vL, vL) + 1e-8
        proj = (dot / norm_sq) * vL
        return proj

    def payoff_gradients(self, vL, vJ):
        # Simplified game gradient:
        # L maximizes norm(vL)
        # J minimizes difference from vL
        grad_L = vL
        grad_J = -(vJ - vL)
        return grad_L, grad_J

    def forward(self, vL, vJ, steps=500):

        vL = vL.clone().to(self.device)
        vJ = vJ.clone().to(self.device)

        # Lorenz initial state
        x = torch.tensor(1.0, device=self.device)
        y = torch.tensor(1.0, device=self.device)
        z = torch.tensor(1.0, device=self.device)

        traj_L = []
        traj_J = []
        traj_z = []

        for _ in range(steps):

            # --- Lorenz system ---
            dx = self.sigma * (y - x)
            dy = x * (self.rho - z) - y
            dz = x * y - self.lorenz_beta * z

            x = x + self.dt * dx
            y = y + self.dt * dy
            z = z + self.dt * dz

            # Modulated cognitive coupling
            alpha = self.alpha0 + self.gamma * z

            # --- Conceptual intersection ---
            I = self.intersection(vL, vJ)

            # --- Game gradients ---
            gL, gJ = self.payoff_gradients(vL, vJ)

            # --- Cognitive ODE ---
            dvL = alpha * (I - vL) + self.beta_game * gL
            dvJ = alpha * (I - vJ) + self.beta_game * gJ

            vL = vL + self.dt * dvL
            vJ = vJ + self.dt * dvJ

            traj_L.append(vL)
            traj_J.append(vJ)
            traj_z.append(z)

        return {
            "vL_traj": torch.stack(traj_L),
            "vJ_traj": torch.stack(traj_J),
            "lorenz_z": torch.stack(traj_z)
        }


def run_simulation():
    dim = 64
    torch.manual_seed(42)

    # Random initial embeddings
    vL0 = torch.randn(dim)
    vJ0 = torch.randn(dim)

    model = CognitiveLorenzField(dim=dim)

    print("--- Cognitive Lorenz Field Simulation ---")
    output = model(vL0, vJ0, steps=1000)

    final_dist = torch.norm(output["vL_traj"][-1] - output["vJ_traj"][-1]).item()
    print(f"Initial distance: {torch.norm(vL0 - vJ0).item():.4f}")
    print(f"Final distance: {final_dist:.4f}")
    print("Simulation Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cognitive Lorenz Game Field Simulation")
    parser.add_argument("--simulate", action="store_true", help="Run the field simulation")
    args = parser.parse_args()

    if args.simulate or not any(vars(args).values()):
        run_simulation()
