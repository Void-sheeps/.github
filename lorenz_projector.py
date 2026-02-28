import torch
import torch.nn as nn
import argparse

# ==========================================================
# 1) SIMULAÇÃO DO SISTEMA DE LORENZ
# ==========================================================

def lorenz_step(state, sigma=10.0, rho=28.0, beta=8/3, dt=0.01):
    x, y, z = state

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return state + dt * torch.tensor([dx, dy, dz])


def simulate_lorenz(steps=10000):
    state = torch.tensor([1.0, 1.0, 1.0])
    trajectory = []

    for _ in range(steps):
        state = lorenz_step(state)
        trajectory.append(state)

    return torch.stack(trajectory)


# ==========================================================
# 3) MÓDULO PARA INTEGRAL SOBRE A ÁREA CAÓTICA
# ==========================================================

class LorenzIntegralProjector(nn.Module):
    def __init__(self):
        super().__init__()

        # Função f_theta(x,y) parametrizada
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, points):
        return self.mlp(points)

    def monte_carlo_integral(self, points):
        """
        Estima:
            I ≈ (1/N) Σ f(x_i, y_i)
        usando a órbita como amostragem da medida invariante
        """
        values = self.forward(points)
        integral_estimate = values.mean()
        return integral_estimate


def run_simulation(steps=15000):
    print(f"Simulating Lorenz attractor with {steps} steps...")
    trajectory = simulate_lorenz(steps=steps)
    xy_plane = trajectory[:, :2]  # projeção no plano (x,y)

    model = LorenzIntegralProjector()
    integral_estimate = model.monte_carlo_integral(xy_plane)

    print("--- Lorenz Integral Projection ---")
    print("Estimated integral over the attractor (Monte Carlo):", integral_estimate.item())

    return trajectory, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lorenz Integral Projector")
    parser.add_argument("--simulate", action="store_true", help="Run simulation")
    parser.add_argument("--steps", type=int, default=15000, help="Simulation steps")
    args = parser.parse_args()

    if args.simulate or not any(vars(args).values()):
        run_simulation(args.steps)
