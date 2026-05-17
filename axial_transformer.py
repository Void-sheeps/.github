from dataclasses import dataclass
from typing import Callable, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CORE STRUCTURES (STRUCTURED DOMAIN)
# ============================================================

@dataclass
class Domain:
    state: torch.Tensor  # structured latent field


@dataclass
class PerturbationOperator:
    apply: Callable[[torch.Tensor], torch.Tensor]
    embedding: torch.Tensor  # learned or fixed operator identity


@dataclass
class DescriptionOperator:
    describe: Callable[[torch.Tensor], torch.Tensor]


# ============================================================
# AXIAL FIELD TRANSFORMER
# ============================================================

class PerturbationFieldTransformer(nn.Module):
    """
    Attention over operator embeddings + base weights + context.
    """

    def __init__(self, d_model=64, n_heads=4):
        super().__init__()

        self.proj_in = nn.Linear(1 + d_model, d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        self.out = nn.Linear(d_model, 1)

    def forward(self, weights, op_embeddings):
        # weights: [n]
        # op_embeddings: [n, d_model]

        w = weights.unsqueeze(-1)
        x = torch.cat([w, op_embeddings], dim=-1)

        x = self.proj_in(x)

        x, _ = self.attn(x, x, x)

        return self.out(x).squeeze(-1)


# ============================================================
# TOKEN
# ============================================================

@dataclass
class PerturbationToken:
    operator: PerturbationOperator
    base_weight: float = 1.0


# ============================================================
# AXIAL SYSTEM (COUPLED FIELD VERSION)
# ============================================================

class AxialTransformerSystem:

    def __init__(self, description: DescriptionOperator, d_model=64):
        self.tokens: List[PerturbationToken] = []
        self.description = description
        self.transformer = PerturbationFieldTransformer(d_model)

    def register(self, op: PerturbationOperator, weight=1.0):
        self.tokens.append(PerturbationToken(op, weight))
        return self

    def _compute_weights(self):
        device = next(self.transformer.parameters()).device
        w = torch.tensor([t.base_weight for t in self.tokens],
                         dtype=torch.float32, device=device)

        emb = torch.stack([t.operator.embedding for t in self.tokens]).to(device)

        w = self.transformer(w, emb)
        return F.softmax(w, dim=0)

    def apply(self, domain: Domain):

        if not self.tokens:
            return {
                "state": domain.state,
                "description": self.description.describe(domain.state),
                "weights": [],
                "axial_relation": "coupled field superposition (empty)"
            }

        weights = self._compute_weights()

        state = domain.state

        # continuous mixture (field superposition approximation)
        new_state = torch.zeros_like(state)

        for token, w in zip(self.tokens, weights):
            transformed = token.operator.apply(state)
            new_state = new_state + w * transformed

        new_domain = Domain(state=new_state)

        return {
            "state": new_domain.state,
            "description": self.description.describe(new_domain.state),
            "weights": weights.detach().cpu().tolist(),
            "axial_relation": "coupled field superposition"
        }
