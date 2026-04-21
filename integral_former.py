import torch
import torch.nn as nn
import math

class IntegralFormer(nn.Module):
    """
    IntegralFormer: Attention as a Kernel Integral Operator
    =======================================================
    Y(i) = X(i) + ∫ K(i,t) V(t) dt

    - No softmax normalization
    - Kernel defined via continuous metric (Gaussian)
    - Explicit quadrature step (dt)
    """
    def __init__(self, seq_len=8, d_model=16, vocab_size=10):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Embedding layer (token + position)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(seq_len, d_model))

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Integration step (Δt)
        self.dt = 1.0 / seq_len

    def get_kernel(self, X):
        """
        Computes the pairwise Gaussian kernel matrix.
        X: (B, N, D)
        Returns: (B, N, N)
        """
        Q = self.W_q(X)
        K = self.W_k(X)

        scale = 1.0 / math.sqrt(self.d_model)

        # Pairwise distance in feature space
        diff = Q.unsqueeze(2) - K.unsqueeze(1)      # (B, N, N, D)
        dist2 = torch.sum(diff ** 2, dim=-1)        # (B, N, N)

        K_kernel = torch.exp(-dist2 * scale)        # (B, N, N)
        return K_kernel

    def forward(self, x):
        """
        x: (batch_size, seq_len) long tensor of token indices
        """
        # --- EMBEDDING ---
        X = self.token_emb(x) + self.pos_emb  # (B, N, D)

        # --- PROJECTIONS ---
        V = self.W_v(X)

        # --- CONTINUOUS KERNEL CONSTRUCTION ---
        K_kernel = self.get_kernel(X)

        # --- INTEGRAL APPROXIMATION ---
        integral = torch.matmul(K_kernel, V) * self.dt  # (B, N, D)

        # --- RESIDUAL CONNECTION ---
        Y = X + integral

        return Y
