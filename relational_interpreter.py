from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple, FrozenSet

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# PARALLELISM REGIMES
# ============================================================================

class ParallelismRegime(Enum):
    """
    PARALLEL:
        delta_norm below the parallel threshold and angular deformation
        within bounds.

    NON_PARALLEL:
        delta_norm above the threshold, or angular deformation exceeds the
        angular threshold despite metric proximity.

    UNVERIFIABLE:
        delta_norm falls within the uncertainty band around the parallel
        threshold. The system cannot assert either regime. This is an
        epistemic band around the decision boundary, not a region of
        numerical identity.
    """

    PARALLEL      = "parallel"
    NON_PARALLEL  = "non_parallel"
    UNVERIFIABLE  = "unverifiable"


# ============================================================================
# HYPERPARAMETERS
# ============================================================================

DIM        = 128
VOCAB_SIZE = 8_192
NUM_HEADS  = 4


# ============================================================================
# RELATIONAL NODE
# ============================================================================

import networkx as nx

class Node(nn.Module):
    """
    Relational-computational node.

    Each node is simultaneously:

        - symbolic       (named, described)
        - topological    (children, relations)
        - differentiable (op: nn.Module)
        - operational    (forward pass)
    """

    def __init__(
        self,
        name:        str,
        description: str                 = "",
        op:          Optional[nn.Module] = None,
        formal_det:  float               = 0.0,
        informal_det: float               = 0.0,
        formal_indet: float               = 0.0,
        informal_indet: float             = 0.0,
    ):
        super().__init__()
        self.node_name   = name
        self.description = description
        self.children:  List[Node]            = []
        self.relations: Dict[str, List[Node]] = {}
        self.op = op if op is not None else nn.Identity()

        # Quadripartitioned determinacy profile
        self.formal_det = formal_det
        self.informal_det = informal_det
        self.formal_indet = formal_indet
        self.informal_indet = informal_indet

    def add_child(self, node: Node) -> Node:
        self.children.append(node)
        self.add_module(f"{self.node_name}_{node.node_name}", node)
        return self

    def relate(self, relation: str, node: Node) -> Node:
        self.relations.setdefault(relation, []).append(node)
        return self

    def traverse(self, depth: int = 0, seen: Optional[set] = None) -> None:
        if seen is None:
            seen = set()
        if id(self) in seen:
            return
        seen.add(id(self))
        indent = "  " * depth
        desc   = f": {self.description}" if self.description else ""
        print(f"{indent}[{self.node_name}]{desc}")
        for child in self.children:
            child.traverse(depth + 1, seen)
        for rel, nodes in self.relations.items():
            for node in nodes:
                print(f"{indent}  --{rel}--> [{node.node_name}]")

    def to_graph(self, G: Optional[nx.DiGraph] = None, seen: set = None) -> nx.DiGraph:
        if G is None:
            G = nx.DiGraph()
        if seen is None:
            seen = set()
        if id(self) in seen:
            return G
        seen.add(id(self))

        G.add_node(self.node_name, description=self.description,
                   formal_det=self.formal_det, informal_det=self.informal_det,
                   formal_indet=self.formal_indet, informal_indet=self.informal_indet)

        for child in self.children:
            G.add_edge(self.node_name, child.node_name, relation="child")
            child.to_graph(G, seen)

        for rel, nodes in self.relations.items():
            for node in nodes:
                # Some relational targets might not be Nodes (e.g. they might be GNode or TriadicSystem)
                # Ensure they have node_name and to_graph
                target_name = getattr(node, "node_name", str(node))
                G.add_edge(self.node_name, target_name, relation=rel)
                if hasattr(node, "to_graph"):
                    node.to_graph(G, seen)

        return G

    def get_role(self, G: nx.DiGraph) -> str:
        """
        Determines if a unit's role is constitutive or cosmetic.
        Constitutive: High formal/informal determinacy and high centrality.
        Cosmetic: High formal/informal indeterminacy or low centrality.
        """
        try:
            centrality = nx.degree_centrality(G)[self.node_name]
        except KeyError:
            centrality = 0.0

        determinacy = (self.formal_det + self.informal_det) / 2
        indeterminacy = (self.formal_indet + self.informal_indet) / 2

        # Weighted score: (Det - Indet) * Centrality
        score = (determinacy - indeterminacy) * (1 + centrality)

        return "constitutive" if score > 0.5 else "cosmetic"

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.op(*args, **kwargs)


# ============================================================================
# STRUCTURAL VALIDATION
# ============================================================================

class StructuralTypeError(TypeError):
    pass


class StructuralValidator:

    @staticmethod
    def validate_numeric(value: object) -> None:
        if isinstance(value, str):
            raise StructuralTypeError(
                "String values are not supported in structural operations."
            )

    @staticmethod
    def validate_tensor(tensor: torch.Tensor) -> None:
        if not tensor.is_floating_point():
            raise StructuralTypeError(
                "Tensor must use a floating-point dtype."
            )


# ============================================================================
# STRUCTURAL SIGNATURE
# ============================================================================

@dataclass(frozen=True)
class StructuralSignature:
    density_scale: float
    irregularity: float
    digest: str

    @staticmethod
    def derive(
        density_scale: float,
        irregularity: float,
        vector: torch.Tensor,
    ) -> StructuralSignature:

        raw = struct.pack("ff", density_scale, irregularity)
        raw += vector.detach().cpu().numpy().tobytes()

        digest = hashlib.sha256(raw).hexdigest()[:16]

        return StructuralSignature(
            density_scale=density_scale,
            irregularity=irregularity,
            digest=digest,
        )

    def __repr__(self) -> str:
        return (
            f"StructuralSignature("
            f"d={self.density_scale:.3f}, "
            f"i={self.irregularity:.3f}, "
            f"#{self.digest})"
        )


# ============================================================================
# UNIT
# ============================================================================

class Unit:

    def __init__(
        self,
        density_scale: float,
        irregularity: float,
        dimension: int,
        *,
        vector: torch.Tensor | None = None,
    ) -> None:

        StructuralValidator.validate_numeric(density_scale)
        StructuralValidator.validate_numeric(irregularity)

        if vector is None:
            base = torch.randn(dimension)
            noise = irregularity * torch.randn(dimension)
            vector = density_scale * (base + noise)

        self._vector = vector.float()

        StructuralValidator.validate_tensor(self._vector)

        self.signature = StructuralSignature.derive(
            density_scale=density_scale,
            irregularity=irregularity,
            vector=self._vector,
        )

    # ------------------------------------------------------------------

    def vector(self) -> torch.Tensor:
        return self._vector.clone()

    def density(self) -> float:
        return self._vector.norm().item()

    def __hash__(self) -> int:
        return hash(self.signature.digest)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented

        return self.signature.digest == other.signature.digest

    def __repr__(self) -> str:
        return (
            f"Unit("
            f"signature={self.signature}, "
            f"density={self.density():.6f})"
        )


# ============================================================================
# FIELD
# ============================================================================

class Field:

    def __init__(self) -> None:
        self._units: list[Unit] = []

    # ------------------------------------------------------------------

    def insert(self, unit: Unit) -> None:
        self._units.append(unit)

    def units(self) -> Tuple[Unit, ...]:
        return tuple(self._units)

    def centroid(self) -> torch.Tensor:
        if not self._units:
            raise ValueError("Field is empty.")

        return torch.stack(
            [unit.vector() for unit in self._units]
        ).mean(dim=0)

    def density(self) -> float:
        if not self._units:
            return 0.0

        return sum(
            unit.density() for unit in self._units
        ) / len(self._units)

    def signatures(self) -> FrozenSet[str]:
        return frozenset(
            unit.signature.digest
            for unit in self._units
        )

    def __len__(self) -> int:
        return len(self._units)

    def __repr__(self) -> str:
        return (
            f"Field("
            f"size={len(self._units)}, "
            f"density={self.density():.6f})"
        )


# ============================================================================
# RELATION REGIME
# ============================================================================

class RelationRegime(IntEnum):
    ALIGNED = 0
    OFFSET = 1
    DIVERGENT = 2
    ASYMMETRIC = 3


# ============================================================================
# ORIENTATION
# ============================================================================

@dataclass(frozen=True)
class Orientation:
    leading_index: int
    asymmetry: float


# ============================================================================
# RELATION
# ============================================================================

@dataclass(frozen=True)
class Relation:
    angle: float
    cosine: float
    magnitude: float

    density_left: float
    density_right: float
    density_ratio: float

    orientation: Orientation
    regime: RelationRegime

    left_signatures: FrozenSet[str]
    right_signatures: FrozenSet[str]


# ============================================================================
# STRUCTURAL GEOMETRY
# ============================================================================

class StructuralGeometry:

    def __init__(
        self,
        alignment_threshold: float = 0.90,
        divergence_angle: float = 20.0,
        asymmetry_threshold: float = 2.0,
    ) -> None:

        self._alignment_threshold = alignment_threshold
        self._divergence_angle = divergence_angle
        self._asymmetry_threshold = asymmetry_threshold

    # ------------------------------------------------------------------

    def compare(
        self,
        left: Field,
        right: Field,
    ) -> Relation:

        left_centroid = left.centroid()
        right_centroid = right.centroid()

        # ------------------------------------------------------------------
        # Angular relation
        # ------------------------------------------------------------------

        cosine = F.cosine_similarity(
            left_centroid.unsqueeze(0),
            right_centroid.unsqueeze(0),
        ).item()

        cosine = max(-1.0, min(1.0, cosine))

        angle = math.degrees(math.acos(cosine))

        # ------------------------------------------------------------------
        # Magnitude
        # ------------------------------------------------------------------

        magnitude = (
            left_centroid - right_centroid
        ).norm().item()

        # ------------------------------------------------------------------
        # Density
        # ------------------------------------------------------------------

        density_left = left.density()
        density_right = right.density()

        density_ratio = (
            density_left / density_right
            if density_right != 0.0
            else float("inf")
        )

        asymmetric = (
            density_ratio > self._asymmetry_threshold
            or density_ratio < 1.0 / self._asymmetry_threshold
        )

        # ------------------------------------------------------------------
        # Orientation
        # ------------------------------------------------------------------

        left_norm = left_centroid.norm().item()
        right_norm = right_centroid.norm().item()

        tolerance = 1e-6

        if abs(left_norm - right_norm) < tolerance:
            leading_index = -1
        else:
            leading_index = (
                0 if left_norm > right_norm else 1
            )

        orientation = Orientation(
            leading_index=leading_index,
            asymmetry=abs(left_norm - right_norm),
        )

        # ------------------------------------------------------------------
        # Regime
        # ------------------------------------------------------------------

        if cosine >= self._alignment_threshold and asymmetric:
            regime = RelationRegime.ASYMMETRIC

        elif cosine >= self._alignment_threshold:
            regime = RelationRegime.ALIGNED

        elif angle > self._divergence_angle:
            regime = RelationRegime.DIVERGENT

        else:
            regime = RelationRegime.OFFSET

        return Relation(
            angle=angle,
            cosine=cosine,
            magnitude=magnitude,

            density_left=density_left,
            density_right=density_right,
            density_ratio=density_ratio,

            orientation=orientation,
            regime=regime,

            left_signatures=left.signatures(),
            right_signatures=right.signatures(),
        )


# ============================================================================
# RELATIONAL DOMAIN & GEOMETRY
# ============================================================================

@dataclass
class RelationalDomain:
    """
    D := [O, S]

    A relational domain is the structural relation between:

        O : substrate state      ∈ R^D
        S : interpretive state   ∈ R^D
    """

    O:    torch.Tensor
    S:    torch.Tensor
    name: str = "D"

    def __post_init__(self) -> None:
        if self.O.shape != self.S.shape:
            raise ValueError(
                f"O and S must have the same shape; "
                f"got O={tuple(self.O.shape)}, S={tuple(self.S.shape)}"
            )

    def internal_angle(self) -> torch.Tensor:
        """
        angle(O, S) — internal relational divergence within this domain.
        """
        cosine = F.cosine_similarity(
            self.O.unsqueeze(0) if self.O.dim() == 1 else self.O,
            self.S.unsqueeze(0) if self.S.dim() == 1 else self.S,
            dim=-1,
        ).clamp(-1.0, 1.0)

        return torch.rad2deg(torch.acos(cosine))

    def relational_vector(self) -> torch.Tensor:
        """
        R(D) := concat(O, S, O - S)  ∈ R^{3D}
        """
        return torch.cat([self.O, self.S, self.O - self.S], dim=-1)


@dataclass
class DomainDifference:
    """
    ΔD := relational deformation between D1 and D2.
    """

    D1:        RelationalDomain
    D2:        RelationalDomain
    angle:     torch.Tensor
    magnitude: torch.Tensor  # normalized; dimension-independent
    regime:    ParallelismRegime


class RelationalGeometry:
    """
    Evaluates relational continuity between domains.
    """

    def __init__(
        self,
        parallel_threshold: float = 0.01,
        unverifiable_band:  float = 0.002,
        angular_threshold:  float = 10.0,    # degrees
    ):
        if unverifiable_band >= parallel_threshold:
            raise ValueError(
                f"unverifiable_band ({unverifiable_band}) must be strictly less "
                f"than parallel_threshold ({parallel_threshold})"
            )

        self.parallel_threshold = parallel_threshold
        self.unverifiable_band  = unverifiable_band
        self.angular_threshold  = angular_threshold

    def relational_angle(
        self,
        D1: RelationalDomain,
        D2: RelationalDomain,
    ) -> torch.Tensor:
        R1 = D1.relational_vector()
        R2 = D2.relational_vector()

        cosine = F.cosine_similarity(
            R1.unsqueeze(0) if R1.dim() == 1 else R1,
            R2.unsqueeze(0) if R2.dim() == 1 else R2,
            dim=-1,
        ).clamp(-1.0, 1.0)

        return torch.rad2deg(torch.acos(cosine))

    def delta(
        self,
        D1: RelationalDomain,
        D2: RelationalDomain,
    ) -> DomainDifference:

        R1 = D1.relational_vector()
        R2 = D2.relational_vector()

        # Dimension-independent L2 distance
        raw        = torch.norm(R1 - R2, p=2)
        delta_norm = raw / (R1.shape[-1] ** 0.5)

        angle     = self.relational_angle(D1, D2)

        # Handling potential batching in tensors
        d_val = delta_norm.mean().item() if delta_norm.numel() > 1 else delta_norm.item()
        a_val = angle.mean().item() if angle.numel() > 1 else angle.item()

        lo = self.parallel_threshold - self.unverifiable_band
        hi = self.parallel_threshold + self.unverifiable_band

        if lo <= d_val <= hi:
            regime = ParallelismRegime.UNVERIFIABLE
        elif d_val < lo:
            if a_val > self.angular_threshold:
                regime = ParallelismRegime.NON_PARALLEL
            else:
                regime = ParallelismRegime.PARALLEL
        else:
            regime = ParallelismRegime.NON_PARALLEL

        return DomainDifference(
            D1=D1,
            D2=D2,
            angle=angle,
            magnitude=delta_norm,
            regime=regime,
        )


# ============================================================================
# O — COMPUTATIONAL SUBSTRATE
# ============================================================================

class ONode(Node):
    """O := input_ids [B, T]  →  substrate [B, T, D]"""

    def __init__(self, dim: int = DIM, **kwargs):
        super().__init__("O", "Computational substrate", **kwargs)
        self.add_child(Node("embeddings",        op=nn.Embedding(VOCAB_SIZE, dim)))
        self.add_child(Node("state_projection",  op=nn.Linear(dim, dim)))
        self.add_child(Node("local_coherence",   op=nn.Conv1d(dim, dim, 3, padding=1)))
        self.add_child(Node("normalization",     op=nn.LayerNorm(dim)))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.children[0](input_ids)
        x = self.children[1](x)
        x = self.children[2](x.transpose(1, 2)).transpose(1, 2)
        x = self.children[3](x)
        return x


# ============================================================================
# C — COUPLING FIELD
# ============================================================================

class CNode(Node):
    """C := substrate [B, T, D]  →  constrained [B, T, D]"""

    def __init__(self, dim: int = DIM, **kwargs):
        super().__init__("C", "Coupling constraints", **kwargs)
        self.add_child(Node("alignment",     op=nn.MultiheadAttention(dim, NUM_HEADS, batch_first=True)))
        self.add_child(Node("span",          op=nn.Linear(dim, dim)))
        self.add_child(Node("threshold",     op=nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())))
        self.add_child(Node("cross_binding", op=nn.Bilinear(dim, dim, dim)))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        aligned, _ = self.children[0](x, x, x, attn_mask=attn_mask)
        spans      = self.children[1](aligned)
        gate       = self.children[2](spans)
        gated      = gate * aligned
        B, T, D    = gated.shape
        cross      = self.children[3](gated.reshape(B*T, D), x.reshape(B*T, D)).reshape(B, T, D)
        return cross


# ============================================================================
# S — INTERPRETATION REGIME
# ============================================================================

class SNode(Node):
    """S := constrained [B, T, D]  →  interpreted [B, T, D]"""

    def __init__(self, dim: int = DIM, **kwargs):
        super().__init__("S", "Interpretation regime", **kwargs)
        self.add_child(Node("lexical_lens",      op=nn.Linear(dim, dim)))
        self.add_child(Node("geometric_lens",    op=nn.Linear(dim, dim)))
        self.add_child(Node("probabilistic_lens",op=nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())))
        self.add_child(Node("fusion_mapping",    op=nn.Linear(3 * dim, dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self.children[0](x), self.children[1](x), self.children[2](x)], dim=-1)
        return self.children[3](fused)


# ============================================================================
# TRIADIC SYSTEM
# ============================================================================

class TriadicSystem(nn.Module):
    """
            [O|
            |C|
            |S]

    Atomic triadic unit: O → C → S.

    Implements the same interface as GNode (forward, traverse, node_name)
    so that G can itself be used as a pole in a higher-order coupling.
    """

    def __init__(self, O: ONode, C: CNode, S: SNode, name: str = "T",
                 formal_det: float = 0.0, informal_det: float = 0.0,
                 formal_indet: float = 0.0, informal_indet: float = 0.0):
        super().__init__()
        self.O = O
        self.C = C
        self.S = S
        self.node_name = name

        # Ensure unique names for components within this manifold
        self.O.node_name = f"{name}_{self.O.node_name}"
        self.C.node_name = f"{name}_{self.C.node_name}"
        self.S.node_name = f"{name}_{self.S.node_name}"
        for comp in [self.O, self.C, self.S]:
            for child in comp.children:
                child.node_name = f"{comp.node_name}_{child.node_name}"

        self.description = "Atomic triadic unit"
        self.relations: Dict[str, List[Node]] = {}

        self.formal_det = formal_det
        self.informal_det = informal_det
        self.formal_indet = formal_indet
        self.informal_indet = informal_indet

        self._wire()

    def _wire(self) -> None:
        for c in self.C.children:
            self.O.relate("constrained_by",    c)
            self.S.relate("modulated_by",      c)
        for s in self.S.children:
            self.O.relate("projected_through", s)
        for o in self.O.children:
            self.C.relate("grounded_in",       o)
        for s in self.S.children:
            self.C.relate("interpreted_through", s)

    def traverse(self, depth: int = 0, seen: Optional[set] = None) -> None:
        if seen is None:
            seen = set()
        if id(self) in seen:
            return
        seen.add(id(self))
        indent = "  " * depth
        print(f"{indent}[{self.node_name}]")
        self.O.traverse(depth + 1, seen)
        self.C.traverse(depth + 1, seen)
        self.S.traverse(depth + 1, seen)

    def to_graph(self, G: Optional[nx.DiGraph] = None, seen: set = None) -> nx.DiGraph:
        if G is None:
            G = nx.DiGraph()
        if seen is None:
            seen = set()
        if id(self) in seen:
            return G
        seen.add(id(self))

        G.add_node(self.node_name, description=self.description,
                   formal_det=self.formal_det, informal_det=self.informal_det,
                   formal_indet=self.formal_indet, informal_indet=self.informal_indet)

        G.add_edge(self.node_name, self.O.node_name, relation="substrate")
        G.add_edge(self.node_name, self.C.node_name, relation="coupling")
        G.add_edge(self.node_name, self.S.node_name, relation="regime")

        self.O.to_graph(G, seen)
        self.C.to_graph(G, seen)
        self.S.to_graph(G, seen)

        return G

    def get_role(self, G: nx.DiGraph) -> str:
        try:
            centrality = nx.degree_centrality(G)[self.node_name]
        except KeyError:
            centrality = 0.0
        determinacy = (self.formal_det + self.informal_det) / 2
        indeterminacy = (self.formal_indet + self.informal_indet) / 2
        score = (determinacy - indeterminacy) * (1 + centrality)
        return "constitutive" if score > 0.5 else "cosmetic"

    def get_domain(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> RelationalDomain:
        """
        Captures the domain [O, S] for this triadic manifold.
        """
        O_out = self.O(input_ids)
        S_out = self.S(self.C(O_out, attn_mask))
        return RelationalDomain(O=O_out, S=S_out, name=self.node_name)

    def get_field(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Field:
        """
        Extracts internal states as a structural Field.
        """
        O_out = self.O(input_ids)
        C_out = self.C(O_out, attn_mask)
        S_out = self.S(C_out)

        field = Field()
        # Flatten batch/time for unit insertion
        for out, scale, irr in [(O_out, 1.0, 0.1), (C_out, 1.1, 0.2), (S_out, 1.2, 0.3)]:
            vec = out.mean(dim=(0, 1))
            field.insert(Unit(density_scale=scale, irregularity=irr, dimension=vec.shape[-1], vector=vec))

        return field

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.S(self.C(self.O(input_ids), attn_mask))


# ============================================================================
# CROSS-FIELD ATTENTION
# ============================================================================

class CrossFieldAttention(nn.Module):
    """
    Attends from one field into another without merging them.

    Each field queries the other; neither is projected into the other's
    internal topology.  The interaction produces a residual correction
    rather than a replacement.

        field_a  ──query──►  field_b  ──►  delta_a
        field_b  ──query──►  field_a  ──►  delta_b

    The correction is gated so the coupling strength is learned, not fixed.
    """

    def __init__(self, dim: int = DIM, num_heads: int = NUM_HEADS):
        super().__init__()
        self.attend_a_to_b = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attend_b_to_a = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate_a        = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.gate_b        = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.norm_a        = nn.LayerNorm(dim)
        self.norm_b        = nn.LayerNorm(dim)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        a, b : [B, T, D]
        returns: a', b'  —  each field corrected by the other, residually
        """
        delta_a, _ = self.attend_a_to_b(a, b, b)   # a queries b
        delta_b, _ = self.attend_b_to_a(b, a, a)   # b queries a

        a_prime = self.norm_a(a + self.gate_a(delta_a) * delta_a)
        b_prime = self.norm_b(b + self.gate_b(delta_b) * delta_b)

        return a_prime, b_prime


# ============================================================================
# G — SYNTHETIC RELATIONAL FIELD
# ============================================================================

class GNode(nn.Module):
    """
     [O|     [O|
     |C|  X  |C|   =   [G]
     |S]     |S]

    G is the emergent field generated by the interaction of two triadic
    manifolds.

    Coupling mechanism:
        1. Each system computes its own field independently.
        2. CrossFieldAttention lets each field query the other,
           producing residual corrections rather than a merger.
        3. The two corrected fields are fused via a learned convex
           combination (not concatenation), preserving their structural
           distinctness in the output.

    G implements the same interface as TriadicSystem so it can serve as
    a pole in a higher-order GNode.
    """

    def __init__(
        self,
        left:  nn.Module,
        right: nn.Module,
        dim:   int = DIM,
        name:  str = "G",
        formal_det: float = 0.0, informal_det: float = 0.0,
        formal_indet: float = 0.0, informal_indet: float = 0.0
    ):
        super().__init__()

        self.left  = left
        self.right = right

        self.node_name = name
        self.description = "Emergent relational field"
        self.relations: Dict[str, List] = {}

        self.formal_det = formal_det
        self.informal_det = informal_det
        self.formal_indet = formal_indet
        self.informal_indet = informal_indet

        # cross-field interaction: fields query each other, residually
        self.cross_attention = CrossFieldAttention(dim)

        # learned convex combination weight (per position, per feature)
        self.mix = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

        self.field_norm = nn.LayerNorm(dim)

        self._wire()

    # ----------------------------------------------------------------------
    # Relational topology
    # ----------------------------------------------------------------------

    def _wire(self) -> None:
        self.relations.setdefault("left_pole",  []).append(self.left)
        self.relations.setdefault("right_pole", []).append(self.right)
        self.relations.setdefault("coupled_via", []).append(
            Node("cross_field_attention")
        )

    def relate(self, relation: str, node: Node) -> GNode:
        self.relations.setdefault(relation, []).append(node)
        return self

    # ----------------------------------------------------------------------
    # Traversal
    # ----------------------------------------------------------------------

    def traverse(self, depth: int = 0, seen: Optional[set] = None) -> None:
        if seen is None:
            seen = set()
        if id(self) in seen:
            return
        seen.add(id(self))
        indent = "  " * depth
        print(f"{indent}[{self.node_name}]: emergent relational field")
        if hasattr(self.left, "traverse"):
            self.left.traverse(depth + 1, seen)
        if hasattr(self.right, "traverse"):
            self.right.traverse(depth + 1, seen)
        for rel, nodes in self.relations.items():
            for n in nodes:
                name = getattr(n, "node_name", str(n))
                print(f"{indent}  --{rel}--> [{name}]")

    def to_graph(self, G: Optional[nx.DiGraph] = None, seen: set = None) -> nx.DiGraph:
        if G is None:
            G = nx.DiGraph()
        if seen is None:
            seen = set()
        if id(self) in seen:
            return G
        seen.add(id(self))

        G.add_node(self.node_name, description=self.description,
                   formal_det=self.formal_det, informal_det=self.informal_det,
                   formal_indet=self.formal_indet, informal_indet=self.informal_indet)

        G.add_edge(self.node_name, self.left.node_name, relation="left_pole")
        G.add_edge(self.node_name, self.right.node_name, relation="right_pole")

        if hasattr(self.left, "to_graph"):
            self.left.to_graph(G, seen)
        if hasattr(self.right, "to_graph"):
            self.right.to_graph(G, seen)

        return G

    def get_role(self, G: nx.DiGraph) -> str:
        try:
            centrality = nx.degree_centrality(G)[self.node_name]
        except KeyError:
            centrality = 0.0
        determinacy = (self.formal_det + self.informal_det) / 2
        indeterminacy = (self.formal_indet + self.informal_indet) / 2
        score = (determinacy - indeterminacy) * (1 + centrality)
        return "constitutive" if score > 0.5 else "cosmetic"

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------

    def get_domain(
        self,
        left_ids:  torch.Tensor,
        right_ids: torch.Tensor,
    ) -> RelationalDomain:
        """
        Captures the domain [O, S] for this emergent field.
        O is the raw ensemble of fields.
        S is the fused result after cross-attention.
        """
        left_field  = self.left(left_ids)
        right_field = self.right(right_ids)

        O_ensemble = (left_field + right_field) / 2

        left_prime, right_prime = self.cross_attention(left_field, right_field)
        alpha = self.mix(torch.cat([left_prime, right_prime], dim=-1))
        fused = alpha * left_prime + (1 - alpha) * right_prime
        S_interpreted = self.field_norm(fused)

        return RelationalDomain(O=O_ensemble, S=S_interpreted, name=self.node_name)

    def get_field(self, left_ids: torch.Tensor, right_ids: torch.Tensor) -> Field:
        """
        Extracts emergent states as a structural Field.
        """
        left_field  = self.left(left_ids)
        right_field = self.right(right_ids)
        left_prime, right_prime = self.cross_attention(left_field, right_field)

        field = Field()
        for out, scale, irr in [(left_field, 1.0, 0.1), (right_field, 1.0, 0.1),
                               (left_prime, 1.1, 0.2), (right_prime, 1.1, 0.2)]:
            vec = out.mean(dim=(0, 1))
            field.insert(Unit(density_scale=scale, irregularity=irr, dimension=vec.shape[-1], vector=vec))

        return field

    def forward(
        self,
        left_ids:  torch.Tensor,
        right_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        left_ids, right_ids : [B, T]
        returns              : [B, T, D]
        """
        left_field  = self.left(left_ids)           # [B, T, D]
        right_field = self.right(right_ids)          # [B, T, D]

        # fields interact without merging: each corrected by the other
        left_prime, right_prime = self.cross_attention(left_field, right_field)

        # learned convex combination
        alpha = self.mix(torch.cat([left_prime, right_prime], dim=-1))  # [B, T, D] ∈ (0,1)
        fused = alpha * left_prime + (1 - alpha) * right_prime          # [B, T, D]

        return self.field_norm(fused)


# ============================================================================
# FIELD STACK
# ============================================================================

class FieldStack(nn.Module):
    """
    Composes a sequence of GNodes into a vertical stack.

    G_0 = couple(T_a, T_b)
    G_1 = couple(G_0, T_c)
    G_2 = couple(G_1, T_d)
    ...

    Each level couples the accumulated field with a fresh triadic system,
    deepening the relational structure without recursing into the previous
    level's internal topology.

    At inference:
        - All left_ids are fed to the current accumulated field.
        - Each right_ids is consumed by the corresponding fresh pole.

    Shape contract: [B, T] × [B, T] → [B, T, D] at every level.
    """

    def __init__(self, depth: int = 3, dim: int = DIM):
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be ≥ 1")

        # level 0: both poles are fresh triadic systems
        self.levels: nn.ModuleList = nn.ModuleList()
        g = GNode(
            build_triadic_system("T_a"),
            build_triadic_system("T_b"),
            dim=dim,
            name="G_0",
        )
        self.levels.append(g)

        # levels 1…depth-1: left pole is the previous G, right is fresh
        for i in range(1, depth):
            g = GNode(
                left  = self.levels[-1],
                right = build_triadic_system(f"T_{chr(ord('c') + i - 1)}"),
                dim   = dim,
                name  = f"G_{i}",
            )
            self.levels.append(g)

    def traverse(self) -> None:
        print(f"\n[FieldStack] depth={len(self.levels)}")
        # only traverse the outermost G; it recursively covers the rest
        self.levels[-1].traverse(depth=1)

    def forward(
        self,
        id_sequences: List[torch.Tensor],   # one [B, T] per stack level + 1
    ) -> torch.Tensor:
        """
        id_sequences : list of length (depth + 1)
            id_sequences[0]        → left pole of G_0
            id_sequences[1]        → right pole of G_0 / left of G_1
            id_sequences[k]        → right pole of G_{k-1}
            id_sequences[-1]       → right pole of G_{depth-1}

        This ensures every triadic system receives its own input, so no
        field absorbs another's token sequence into its own embedding space.
        """
        if len(id_sequences) != len(self.levels) + 1:
            raise ValueError(
                f"FieldStack of depth {len(self.levels)} requires "
                f"{len(self.levels) + 1} id_sequences; got {len(id_sequences)}"
            )

        # level 0 consumes the first two sequences
        accumulated = self.levels[0](id_sequences[0], id_sequences[1])

        # each subsequent level couples accumulated field with next sequence
        for i, g in enumerate(self.levels[1:], start=1):
            # g.left IS self.levels[i-1], so we pass accumulated as left_ids
            # but g.left is a GNode that expects two id tensors...
            # solution: wrap the accumulated tensor in a PassthroughTriad
            accumulated = _couple_with_field(g, accumulated, id_sequences[i + 1])

        return accumulated


def _couple_with_field(
    g:           GNode,
    left_tensor: torch.Tensor,
    right_ids:   torch.Tensor,
) -> torch.Tensor:
    """
    When g.left is itself a GNode already evaluated, bypass its forward
    and inject the precomputed left_tensor directly into the coupling.

    This avoids re-running previous levels on potentially different inputs.
    """
    right_field = g.right(right_ids)

    left_prime, right_prime = g.cross_attention(left_tensor, right_field)

    alpha = g.mix(torch.cat([left_prime, right_prime], dim=-1))
    fused = alpha * left_prime + (1 - alpha) * right_prime

    return g.field_norm(fused)


# ============================================================================
# FACTORY
# ============================================================================

def build_triadic_system(name: str = "T", dim: int = DIM) -> TriadicSystem:
    return TriadicSystem(ONode(dim), CNode(dim), SNode(dim), name=name)


# ============================================================================
# ENTRYPOINT
# ============================================================================

if __name__ == "__main__":

    torch.manual_seed(0)

    B, T = 2, 16

    def rand_ids() -> torch.Tensor:
        return torch.randint(0, VOCAB_SIZE, (B, T))

    # ── single-level G ───────────────────────────────────────────────────────

    T1 = build_triadic_system("T1")
    T2 = build_triadic_system("T2")
    G  = GNode(T1, T2)

    print("\n── SINGLE-LEVEL G ──")
    G.traverse()

    out = G(rand_ids(), rand_ids())
    print(f"\nG output: {tuple(out.shape)}")

    params_G = sum(p.numel() for p in G.parameters())
    print(f"G params: {params_G:,}")

    # ── field stack: G_0 = T×T, G_1 = G_0×T, G_2 = G_1×T ──────────────────

    stack = FieldStack(depth=3)

    print("\n\n── FIELD STACK (depth=3) ──")
    stack.traverse()

    # stack of depth 3 consumes 4 id sequences
    sequences = [rand_ids() for _ in range(4)]
    out_stack = stack(sequences)
    print(f"\nstack output: {tuple(out_stack.shape)}")

    params_stack = sum(p.numel() for p in stack.parameters())
    print(f"stack params: {params_stack:,}")

    # ── parameter breakdown ──────────────────────────────────────────────────

    print("\n── PARAMETER BREAKDOWN ──\n")
    for name, mod in [("G (single)", G), ("FieldStack (depth=3)", stack)]:
        total = sum(p.numel() for p in mod.parameters())
        print(f"  {name:<28} {total:>12,}")
