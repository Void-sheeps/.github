from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ONTOLOGY
# ============================================================

@dataclass
class Domain:
    state: torch.Tensor  # [B, ...]


@dataclass
class PerturbationOperator:
    apply: Callable[[torch.Tensor], torch.Tensor]


@dataclass
class DescriptionOperator:
    describe: Callable[[torch.Tensor], Any]


@dataclass
class PerturbationToken:
    operator: PerturbationOperator
    embedding: torch.Tensor  # [D]


# ============================================================
# ROUTER (UNIFIED FIELD MECHANISM)
# ============================================================

class AxialRouter(nn.Module):
    """
    Produces alpha_i(x): mixture coefficients over operators.

    This replaces:
      - discrete selection
      - field weighting
      - dual regimes

    Everything becomes a differentiable measure over operator space.
    """

    def __init__(self, d_model: int, n_heads: int = 4, temperature: float = 1.0):
        super().__init__()

        self.temperature = temperature

        self.state_proj = nn.Linear(1, d_model)
        self.op_proj = nn.Linear(d_model, d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        self.out = nn.Linear(d_model, 1)

    def forward(self, weights, op_embeds):
        """
        weights:   [B, N]
        op_embeds: [B, N, D]
        """

        w = self.state_proj(weights.unsqueeze(-1))
        o = self.op_proj(op_embeds)

        x = w + o

        x, _ = self.attn(x, x, x)

        logits = self.out(x).squeeze(-1)

        # temperature-controlled mixture
        logits = logits / max(self.temperature, 1e-6)

        alpha = F.softmax(logits, dim=-1)

        return alpha


# ============================================================
# AXIAL SYSTEM (UNIFIED FORMULATION)
# ============================================================

class AxialTransformerSystem(nn.Module):

    def __init__(
        self,
        description: DescriptionOperator,
        d_model: int = 64,
        temperature: float = 1.0
    ):
        super().__init__()

        self.tokens: List[PerturbationToken] = []
        self.description = description

        self.d_model = d_model

        self.router = AxialRouter(d_model, temperature=temperature)

    def register(self, op: PerturbationOperator, embedding: torch.Tensor):
        assert embedding.shape[-1] == self.d_model

        self.tokens.append(
            PerturbationToken(op, embedding)
        )

        return self

    # ========================================================
    # EMBEDDING STACK
    # ========================================================

    def _get_embeddings(self, device=None):
        embs = []

        for t in self.tokens:
            embs.append(t.embedding)

        return torch.stack(embs).to(device)  # [N, D]

    # ========================================================
    # UNIFIED DYNAMICS
    # ========================================================

    def apply(
        self,
        domain: Domain,
        base_weights: torch.Tensor,
        hard: bool = False,          # discrete limit
        gumbel_tau: Optional[float] = None
    ) -> Dict[str, Any]:

        state = domain.state
        B = state.shape[0]
        N = len(self.tokens)

        if N == 0:
            return {
                "state": state,
                "description": self.description.describe(state),
                "weights": None,
                "alpha": None
            }

        op_embeds = self._get_embeddings(state.device)
        op_embeds = op_embeds.unsqueeze(0).expand(B, -1, -1)

        alpha = self.router(base_weights, op_embeds)  # [B, N]

        # ====================================================
        # OPTIONAL DISCRETE LIMIT (Gumbel-Softmax view)
        # ====================================================
        if hard:
            # straight-through argmax
            idx = alpha.argmax(dim=-1)
            alpha_hard = torch.zeros_like(alpha)
            alpha_hard.scatter_(1, idx.unsqueeze(-1), 1.0)
            alpha = (alpha_hard - alpha).detach() + alpha

        elif gumbel_tau is not None:
            g = F.gumbel_softmax(
                torch.log(alpha + 1e-8),
                tau=gumbel_tau,
                hard=False
            )
            alpha = g

        # ====================================================
        # FIELD COMPOSITION (CORE EQUATION)
        # ====================================================

        new_state = torch.zeros_like(state)

        for i, token in enumerate(self.tokens):

            transformed = token.operator.apply(state)

            w = alpha[:, i].view(
                B,
                *([1] * (state.dim() - 1))
            )

            new_state = new_state + w * transformed

        return {
            "state": new_state,
            "description": self.description.describe(new_state),
            "alpha": alpha,
            "mode": "unified_field",
            "interpretation": "T(x) = Σ α_i(x) T_i(x)"
        }


# ============================================================
# AXIAL FIELD AXIOMS (UPDATED)
# ============================================================

AXIAL_FIELD_AXIOMS = {
    "operators": "basis functions in learned function space",
    "alpha": "state-conditioned measure over operator manifold",
    "domain": "latent tensor field",
    "dynamics": "single mixture operator (no regime split)",
    "core_equation": "T(x) = Σ α_i(x) T_i(x)",
    "discrete_limit": "Gumbel-Softmax / argmax degeneration of measure",
}
