from dataclasses import dataclass
from typing import Callable, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ONTOLOGY
# ============================================================

@dataclass
class Domain:
    """
    Estado não estruturado (versão original),
    mas aqui tratado como tensor latente genérico.
    """
    state: torch.Tensor  # [B, ...]


@dataclass
class PerturbationOperator:
    """
    apply: Domain → Domain
    versão discreta (pipeline interpretável)
    """
    apply: Callable[[torch.Tensor], torch.Tensor]


@dataclass
class DescriptionOperator:
    """
    Domain → str / latent description
    """
    describe: Callable[[torch.Tensor], Any]


@dataclass
class PerturbationToken:
    operator: PerturbationOperator
    base_weight: float = 1.0
    embedding: torch.Tensor = None  # identity opcional do operador


# ============================================================
# REWEIGHTING MECHANISM (TRANSFORMER CORE)
# ============================================================

class PerturbationTransformer(nn.Module):
    """
    token-token coupling (original regime)
    """
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()

        self.embed_w = nn.Linear(1, d_model)
        self.embed_op = nn.Linear(d_model, d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        self.out = nn.Linear(d_model, 1)

    def forward(self, weights, op_embeds):
        # weights: [B, N]
        # op_embeds: [B, N, D]

        w = self.embed_w(weights.unsqueeze(-1))
        o = self.embed_op(op_embeds)

        x = w + o

        x, _ = self.attn(x, x, x)

        return self.out(x).squeeze(-1)


# ============================================================
# COUPLING MECHANISM (GALERKIN FIELD TRANSFORMER)
# ============================================================

class GalerkinPerturbationTransformer(nn.Module):
    """
    cross-operator interference (field regime)
    """

    def __init__(self, d_model=64, n_heads=4):
        super().__init__()

        self.w_proj = nn.Linear(1, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        self.out = nn.Linear(d_model, 1)

    def forward(self, weights, op_embeds):
        # weights: [B, N]
        # op_embeds: [B, N, D]

        w = self.w_proj(weights.unsqueeze(-1))
        o = self.o_proj(op_embeds)

        x = w + o

        x, _ = self.attn(x, x, x)

        return self.out(x).squeeze(-1)


# ============================================================
# AXIAL SYSTEM (HYBRID DYNAMICS)
# ============================================================

class AxialTransformerSystem:

    def __init__(
        self,
        description: DescriptionOperator,
        d_model: int = 64,
        mode: str = "field"  # "discrete" | "field"
    ):
        self.tokens: List[PerturbationToken] = []
        self.description = description
        self.d_model = d_model
        self.mode = mode

        self.discrete_reweighter = PerturbationTransformer(d_model)
        self.field_reweighter = GalerkinPerturbationTransformer(d_model)

    def register(self, op: PerturbationOperator, weight=1.0, embedding=None):
        self.tokens.append(
            PerturbationToken(op, weight, embedding)
        )
        return self

    # ========================================================
    # WEIGHT SPACE (scalar → tensor fields)
    # ========================================================

    def _get_embeddings(self):
        embs = []
        for t in self.tokens:
            if t.embedding is None:
                # fallback: degenerate embedding
                embs.append(torch.zeros(self.d_model))
            else:
                embs.append(t.embedding)
        return torch.stack(embs)

    def _compute_weights(self, base_weights):
        op_embeds = self._get_embeddings().unsqueeze(0).expand(
            base_weights.shape[0], -1, -1
        )

        if self.mode == "discrete":
            logits = self.discrete_reweighter(base_weights, op_embeds)
        else:
            logits = self.field_reweighter(base_weights, op_embeds)

        return F.softmax(logits, dim=-1)

    # ========================================================
    # DYNAMICS (SEQUENTIAL vs SUPERPOSITION)
    # ========================================================

    def apply(self, domain: Domain, base_weights: torch.Tensor) -> Dict[str, Any]:

        if not self.tokens:
            return {
                "state": domain.state,
                "description": self.description.describe(domain.state),
                "weights": None,
                "mode": self.mode
            }

        weights = self._compute_weights(base_weights)

        state = domain.state

        # ====================================================
        # DISCRETE REGIME (SEQUENTIAL / ORDER-DEPENDENT)
        # ====================================================
        if self.mode == "discrete":

            current = state

            for i, t in enumerate(self.tokens):
                # NOTE: w is computed but not used to scale the operator application in discrete mode in original code
                # only used to gate if > 0.1
                if (weights[:, i] > 0.1).any():
                    current = t.operator.apply(current)

            new_state = current

        # ====================================================
        # FIELD REGIME (SUPERPOSITION / GALERKIN)
        # ====================================================
        else:

            new_state = torch.zeros_like(state)

            for i, t in enumerate(self.tokens):

                transformed = t.operator.apply(state)

                w = weights[:, i].view(
                    weights.shape[0],
                    *([1] * (state.dim() - 1))
                )

                new_state = new_state + w * transformed

        return {
            "state": new_state,
            "description": self.description.describe(new_state),

            # scalar → tensor field
            "weights": weights,

            "mode": self.mode,

            # Axial interpretation preserved
            "axial_relation": "description ↔ transformation (dual field projection)"
        }


# ============================================================
# FIELD INTERPRETATION AXIOMS
# ============================================================

AXIAL_FIELD_AXIOMS = {
    "operators": "basis functions in function space",
    "weights": "measure over operator manifold",
    "domain": "discretized PDE-like state",
    "duality": "description ↔ transformation",
    "regimes": {
        "discrete": "sequential symbolic computation",
        "field": "continuous superposition (Galerkin regime)"
    }
}
