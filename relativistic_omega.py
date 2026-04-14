import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# ----------------------------
# Utility: Random Permutation
# ----------------------------
def random_permute_batch(x):
    """
    Randomly permutes the sequence dimension (L) for each batch element.
    x: (B, L, K)
    """
    B, L, K = x.shape
    perm = torch.stack([torch.randperm(L, device=x.device) for _ in range(B)])
    return x[torch.arange(B, device=x.device).unsqueeze(1), perm]


# ----------------------------
# Meaning Network (M_theta)
# ----------------------------
class MeaningNet(nn.Module):
    """
    Computes a semantic gate between two states based on their
    encoded cosine similarity.
    """
    def __init__(self, L, K, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(L * K, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

    def forward(self, Si, Sj):
        # Flatten and encode
        ei = self.encoder(Si.view(Si.size(0), -1))
        ej = self.encoder(Sj.view(Sj.size(0), -1))

        # Cosine similarity -> gate
        sim = F.cosine_similarity(ei, ej, dim=-1)
        return torch.sigmoid(sim).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)


# ----------------------------
# Exchange Transform (A_theta)
# ----------------------------
class Exchange(nn.Module):
    """
    Applies a learned linear transformation to the feature space.
    """
    def __init__(self, K):
        super().__init__()
        self.A = nn.Parameter(torch.randn(K, K) * 0.1)

    def forward(self, Sj):
        # (B, L, K) @ (K, K) -> (B, L, K)
        return torch.matmul(Sj, self.A)


# ----------------------------
# Relativistic Operator (R_phi)
# ----------------------------
class RelativisticTransform(nn.Module):
    """
    Applies a coordinate-like distortion (rotation/scaling) to the state.
    """
    def __init__(self, K):
        super().__init__()
        self.R = nn.Parameter(torch.eye(K) + 0.01 * torch.randn(K, K))

    def forward(self, S):
        return torch.matmul(S, self.R)


# ----------------------------
# Ω Operator (Relativistic)
# ----------------------------
class RelativisticOmegaOperator(nn.Module):
    """
    Ω Operator
    ==========
    Implements a relativistic interaction model between states including:
    - Meaning-based gating
    - Learned feature exchange
    - Positional scattering (random permutation)
    - Global reinforcement weighting
    """
    def __init__(self, L, K):
        super().__init__()
        self.L = L
        self.K = K

        self.meaning = MeaningNet(L, K)
        self.exchange = Exchange(K)
        self.relativistic = RelativisticTransform(K)

        # Global reinforcement weights
        self.W = nn.Parameter(torch.zeros(K))

    def forward(self, Si, Sj):
        """
        Si, Sj: (B, L, K)
        """
        # 1. Meaning gate
        M = self.meaning(Si, Sj)  # (B,1,1)

        # 2. Exchange
        E = self.exchange(Sj)  # (B,L,K)

        # 3. Random positional scattering
        E_perm = random_permute_batch(E)

        # 4. Interaction
        interaction = M * E_perm

        # 5. Update state
        S_new = Si + interaction

        # 6. Relativistic distortion
        S_new = self.relativistic(S_new)

        # 7. Reinforcement weighting
        W_soft = F.softmax(self.W, dim=0)
        S_new = S_new * W_soft

        # 8. Normalize to simplex
        S_new = F.softmax(S_new, dim=-1)

        return S_new

    def update_reinforcement(self, Si, Sj, alpha=0.01, beta=0.005):
        """
        Reinforcement update based on agreement signal.
        """
        # Agreement signal: Mean element-wise product across batch and sequence
        agreement = (Si * Sj).mean(dim=(0, 1))  # (K,)

        noise = torch.rand_like(agreement) * beta

        with torch.no_grad():
            self.W += alpha * agreement - noise
