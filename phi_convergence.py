#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import math
import argparse

class PhiConvergenceLayer(nn.Module):
    """
    A learnable layer that projects features onto the Phi Bilateral Field.

    It applies the 'Density' metric as a modulation mask.
    - Features at low indices (k=0,1...) are treated as 'Macroscopic' (low density).
    - Features at high indices (k=N) are treated as 'Singularity' (high density).

    The layer learns:
    1. The scale 'f' (Projection amplitude)
    2. The decay rate 'phi' (Geometry), initialized at 1.618...
    """
    def __init__(self, num_features, init_phi=1.61803398875, epsilon=1e-12):
        super().__init__()
        self.num_features = num_features
        self.epsilon = epsilon

        # We treat the feature dimension as the 'k' steps
        self.k = torch.arange(0, num_features, dtype=torch.float32)

        # Learnable Parameters
        # We use Log-space for Phi to ensure it stays positive during training
        self.log_phi = nn.Parameter(torch.log(torch.tensor(init_phi)))

        # Learnable Scale (f)
        self.f = nn.Parameter(torch.tensor(1.0))

        # A linear transformation to mix the inputs before projection
        self.input_mixer = nn.Linear(num_features, num_features)

    def get_field_metrics(self):
        """Reconstructs the field based on current learned parameters."""
        phi = torch.exp(self.log_phi)
        k = self.k.to(phi.device)

        # 1. Projection: P(k) = f * phi^-k
        projection = self.f * torch.pow(phi, -k)

        # 2. Bilateral Branches (Symmetric)
        # We use the absolute projection as the 'width' of the corridor
        width = 2.0 * torch.abs(projection) + self.epsilon

        # 3. Informational Density: D = k / width
        # This creates the 'Attention Curve'
        density = k / width

        # Normalize density to 0-1 range for stable gating (Sigmoid-like behavior)
        # We use a Softmax over the features to create a probability distribution
        # or a Tanh to create a modulation gate. Here we use Tanh.
        gating_signal = torch.tanh(density)

        return gating_signal, phi

    def forward(self, x):
        """
        x: Input tensor of shape (Batch, Num_Features)
        """
        # 1. Mix input signals
        x_mixed = self.input_mixer(x)

        # 2. Calculate the Phi-Field Gating
        gating_signal, _ = self.get_field_metrics()

        # 3. Apply Bilateral Convergence
        # The signal is modulated by the geometric density of the field
        # High 'k' indices (Singularity) get stronger weights if density is high.
        x_projected = x_mixed * gating_signal

        return x_projected

def run_simulation():
    # 1. Setup
    torch.manual_seed(42)
    feature_dim = 64  # This represents 'max_k'
    batch_size = 16

    # Model: A simple network using our Phi Layer
    model = nn.Sequential(
        PhiConvergenceLayer(feature_dim, init_phi=1.618),
        nn.Linear(feature_dim, 1) # Regress to a single value
    )

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print(f"--- Initial State ---")
    print(f"Phi (Golden Ratio): {math.sqrt(5)/2 + 0.5:.6f}")
    current_phi = torch.exp(model[0].log_phi).item()
    print(f"Model Phi Start: {current_phi:.6f}")

    # 2. Synthetic Data Task
    # We want the model to learn that only the 'Deep' features (high k) matter.
    # Target = Sum of last 5 features (Simulating a singularity)
    inputs = torch.randn(batch_size, feature_dim)
    targets = inputs[:, -5:].sum(dim=1, keepdim=True)

    # 3. Training Loop
    print("\n--- Training (Learning to Converge) ---")
    for epoch in range(501):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, targets)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            current_phi = torch.exp(model[0].log_phi).item()
            current_f = model[0].f.item()
            print(f"Epoch {epoch}: Loss {loss.item():.6f} | Learned Phi: {current_phi:.6f} | Scale f: {current_f:.4f}")

    # 4. Analysis of the Learned Field
    print("\n--- Final Field Analysis ---")
    gate_curve, final_phi = model[0].get_field_metrics()

    print(f"Final Phi: {final_phi.item():.6f}")
    print(f"Interpretation: The network adjusted Phi to optimize information density.")

    # Show the gating values for the first few (Macro) vs last few (Micro)
    print("\nGating Weights (Ascension vs Regression):")
    print(f"k=0  (Macro): {gate_curve[0].item():.6f}")
    print(f"k=32 (Mid)  : {gate_curve[32].item():.6f}")
    print(f"k=63 (Micro): {gate_curve[63].item():.6f}")
    print("\nSimulation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phi Convergence Layer Simulation")
    parser.add_argument("--simulate", action="store_true", help="Run the convergence simulation")
    args = parser.parse_args()

    if args.simulate or not any(vars(args).values()):
        run_simulation()
