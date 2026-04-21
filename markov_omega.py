import torch
import torch.nn as nn
import torch.optim as optim
import math

class MarkovOmegaOperator(nn.Module):
    """
    Markov Omega Operator
    =====================
    Implements the Omega Operator using a Markov Chain Monte Carlo (MCMC) estimator.
    Ω[A] = D * exp(iφ) * E_{v~p(v)} [ K(v)A(v) / p(v) ]
    """
    def __init__(self, N=128, T=64, sigma=0.05, device="cpu"):
        super().__init__()

        self.N = N              # discretization points
        self.T = T              # Markov chain length
        self.sigma = sigma      # random walk std
        self.device = device

        # Discretized domain [0,1]
        self.register_buffer("v_grid", torch.linspace(0, 1, N, device=device))

        # Learnable kernel K(v)
        self.K = nn.Parameter(torch.randn(N))

        # Complex scaling parameters
        self.log_D = nn.Parameter(torch.tensor(0.0))
        self.phi = nn.Parameter(torch.tensor(0.0))

    def normalize_kernel(self):
        return self.K / (torch.norm(self.K) + 1e-8)

    def interpolate_A(self, v, A):
        """
        Linear interpolation of A over grid
        """
        idx = v * (self.N - 1)
        i0 = torch.clamp(idx.long(), 0, self.N - 2)
        i1 = i0 + 1

        w = idx - i0.float()
        return (1 - w) * A[i0] + w * A[i1]

    def interpolate_K(self, v, K):
        """
        Linear interpolation of K over grid
        """
        idx = v * (self.N - 1)
        i0 = torch.clamp(idx.long(), 0, self.N - 2)
        i1 = i0 + 1

        w = idx - i0.float()
        return (1 - w) * K[i0] + w * K[i1]

    def markov_chain(self, p=None):
        """
        Generate Markov chain v₁ → ... → v_T
        """
        device = self.v_grid.device
        # Start at a random position
        v = torch.rand(1, device=device)
        samples = []

        for _ in range(self.T):
            # Metropolis-Hastings proposal
            proposal = v + torch.randn_like(v) * self.sigma
            proposal = torch.clamp(proposal, 0.0, 1.0)

            if p is None:
                # Uniform random walk
                accept_prob = 1.0
            else:
                # Importance sampling via p(v)
                accept_prob = torch.minimum(
                    torch.tensor(1.0, device=device),
                    p(proposal) / (p(v) + 1e-8)
                )

            if torch.rand(1, device=device) < accept_prob:
                v = proposal

            samples.append(v)

        return torch.stack(samples)

    def forward(self, A, p=None):
        """
        Compute Ω[A] using Markov estimator
        """
        K = self.normalize_kernel()

        v_samples = self.markov_chain(p)

        rewards = []

        # Iterating over samples (could be vectorized if T is large,
        # but for T=64-128 this is fine and readable)
        for v in v_samples:
            k_val = self.interpolate_K(v, K)
            a_val = self.interpolate_A(v, A)

            r = k_val * a_val

            if p is None:
                # p(v) = 1 (Uniform)
                w = 1.0
            else:
                # Correct for non-uniform sampling density
                w = 1.0 / (p(v) + 1e-8)

            rewards.append(r * w)

        rewards = torch.stack(rewards)
        estimate = rewards.mean()

        # Complex scaling: D * exp(iφ)
        D = torch.exp(self.log_D)

        # Output as a torch.complex64
        real = D * torch.cos(self.phi) * estimate
        imag = D * torch.sin(self.phi) * estimate

        return torch.complex(real, imag)

    def smoothness_loss(self):
        """
        Regularization: penalize sharp transitions in the kernel
        """
        K = self.normalize_kernel()
        return torch.mean((K[1:] - K[:-1])**2)
