from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx


class Relation(Enum):
    LOCAL = "local"
    TOPOLOGICAL = "topological"
    CONTEXTUAL = "contextual"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    JUSTIFICATIVE = "justificative"


class Regime(Enum):
    STRICT = "strict"
    EXPANSIVE = "expansive"


@dataclass(frozen=True)
class Node:
    identifier: str
    content: str
    categories: Tuple[str, ...] = ()


@dataclass
class Edge:
    source: str
    target: str
    relation: Relation
    weight: float
    grad: float = 0.0


@dataclass
class FieldState:
    active_nodes: Dict[str, float] = field(default_factory=dict)
    node_grads: Dict[str, float] = field(default_factory=dict)
    active_edges: List[Edge] = field(default_factory=list)
    contextual_pressure: float = 1.0
    local_rigidity: float = 1.0
    loss: float = 0.0

    def inject(self, node_id: str, magnitude: float) -> None:
        self.active_nodes[node_id] = self.active_nodes.get(node_id, 0.0) + magnitude

    def normalize(self) -> None:
        total = sum(abs(v) for v in self.active_nodes.values())
        if total == 0:
            return
        for key in list(self.active_nodes.keys()):
            self.active_nodes[key] /= total


class SemanticField:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []

    def add_node(self, node: Node) -> None:
        self.nodes[node.identifier] = node

    def connect(
        self,
        source: str,
        target: str,
        relation: Relation,
        weight: float,
    ) -> None:
        self.edges.append(
            Edge(
                source=source,
                target=target,
                relation=relation,
                weight=weight,
            )
        )

    def adjacent(self, node_id: str) -> List[Edge]:
        return [edge for edge in self.edges if edge.source == node_id]

    def zero_grad(self) -> None:
        for edge in self.edges:
            edge.grad = 0.0


class ConstraintResolver:
    def __init__(self, field: SemanticField) -> None:
        self.field = field

    def propagate(
        self,
        anchors: Dict[str, float],
        target_node: str,
        depth: int = 3,
        contextual_gain: float = 1.0,
        local_decay: float = 0.5,
        regime: Regime = Regime.STRICT,
    ) -> FieldState:
        state = FieldState(
            contextual_pressure=contextual_gain,
            local_rigidity=local_decay,
        )

        for node_id, magnitude in anchors.items():
            state.inject(node_id, magnitude)

        execution_tape: List[Tuple[str, str, Edge, float, float]] = []
        frontier = list(anchors.items())
        visited: Set[Tuple[str, int]] = set()

        for step in range(depth):
            next_frontier: List[Tuple[str, float]] = []

            for node_id, energy in frontier:
                marker = (node_id, step)
                if marker in visited:
                    continue
                visited.add(marker)

                for edge in self.field.adjacent(node_id):
                    modifier = self._get_modifier(
                        edge.relation, contextual_gain, local_decay, regime
                    )
                    propagated = energy * edge.weight * modifier

                    if propagated <= 0:
                        continue

                    state.inject(edge.target, propagated)
                    state.active_edges.append(edge)
                    execution_tape.append((node_id, edge.target, edge, energy, modifier))
                    next_frontier.append((edge.target, propagated))

            frontier = next_frontier

        raw_nodes = state.active_nodes.copy()
        state.normalize()

        loss, loss_grads = self._compute_loss(state, target_node)
        state.loss = loss

        self._backpropagate(state, execution_tape, raw_nodes, loss_grads)
        return state

    def _get_modifier(
        self,
        relation: Relation,
        contextual_gain: float,
        local_decay: float,
        regime: Regime,
    ) -> float:
        if regime == Regime.EXPANSIVE:
            return contextual_gain

        if relation in {Relation.TOPOLOGICAL, Relation.CONTEXTUAL, Relation.ANALOGICAL}:
            return contextual_gain
        if relation in {Relation.LOCAL, Relation.CAUSAL, Relation.JUSTIFICATIVE}:
            return local_decay
        return 1.0

    def _compute_loss(
        self,
        state: FieldState,
        target_node: str,
    ) -> Tuple[float, Dict[str, float]]:
        loss = -state.active_nodes.get(target_node, 0.0)
        loss_grads: Dict[str, float] = {node_id: 0.0 for node_id in state.active_nodes}
        if target_node in loss_grads:
            loss_grads[target_node] = -1.0
        return loss, loss_grads

    def _backpropagate(
        self,
        state: FieldState,
        execution_tape: List[Tuple[str, str, Edge, float, float]],
        raw_nodes: Dict[str, float],
        loss_grads: Dict[str, float],
    ) -> None:
        self.field.zero_grad()

        total_raw = sum(abs(v) for v in raw_nodes.values())
        if total_raw == 0:
            return

        for node_id in state.active_nodes:
            state.node_grads[node_id] = loss_grads.get(node_id, 0.0)

        for node_id, norm_val in state.active_nodes.items():
            dL_dnorm = state.node_grads[node_id]
            for target_id, raw_val in raw_nodes.items():
                if node_id == target_id:
                    dnorm_draw = (total_raw - raw_val) / (total_raw**2)
                else:
                    dnorm_draw = -raw_nodes[node_id] / (total_raw**2)

                if target_id not in state.node_grads:
                    state.node_grads[target_id] = 0.0
                state.node_grads[target_id] += dL_dnorm * dnorm_draw

        for source_id, target_id, edge, source_energy, modifier in reversed(
            execution_tape
        ):
            dL_dtarget = state.node_grads.get(target_id, 0.0)
            edge.grad += dL_dtarget * source_energy * modifier
            dL_dsource = dL_dtarget * edge.weight * modifier
            state.node_grads[source_id] = (
                state.node_grads.get(source_id, 0.0) + dL_dsource
            )


def plot_relational_field(
    state: FieldState, field: SemanticField, filename: str, title: Optional[str] = None
) -> None:
    G = nx.DiGraph()
    for node_id, node in field.nodes.items():
        activation = state.active_nodes.get(node_id, 0.0)
        G.add_node(node_id, label=node_id, activation=activation)

    for edge in field.edges:
        G.add_edge(edge.source, edge.target, relation=edge.relation.value)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)

    activations = [G.nodes[node]["activation"] for node in G.nodes()]
    node_sizes = [300 + 5000 * a for a in activations]
    node_colors = activations

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        alpha=0.8,
    )
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    edge_labels = {
        (u, v): d["relation"] for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title or f"Relational Field Activation ({filename})")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max(activations) if activations else 1))
    plt.colorbar(sm, label="Activation Magnitude", ax=plt.gca())
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Visualization saved to {filename}")


def setup_default_field() -> SemanticField:
    field = SemanticField()

    field.add_node(
        Node(
            identifier="absorption_identity",
            content="A + A.B = A",
            categories=("formal", "boolean"),
        )
    )
    field.add_node(
        Node(
            identifier="local_formal_constraint",
            content="strict symbolic interpretation",
            categories=("formal", "constraint"),
        )
    )
    field.add_node(
        Node(
            identifier="topological_continuity",
            content="global relational continuity",
            categories=("topological", "contextual"),
        )
    )
    field.add_node(
        Node(
            identifier="semantic_refinement",
            content="refinement under contextual compatibility",
            categories=("semantic", "relational"),
        )
    )
    field.add_node(
        Node(
            identifier="justificative_displacement",
            content="causal justification incompatible with local inference",
            categories=("causal", "inferential"),
        )
    )

    field.connect(
        source="absorption_identity",
        target="local_formal_constraint",
        relation=Relation.LOCAL,
        weight=1.0,
    )
    field.connect(
        source="absorption_identity",
        target="topological_continuity",
        relation=Relation.TOPOLOGICAL,
        weight=0.9,
    )
    field.connect(
        source="topological_continuity",
        target="semantic_refinement",
        relation=Relation.CONTEXTUAL,
        weight=0.8,
    )
    field.connect(
        source="semantic_refinement",
        target="justificative_displacement",
        relation=Relation.ANALOGICAL,
        weight=0.7,
    )
    return field


def run_simulation(plot: bool = False):
    field = setup_default_field()
    resolver = ConstraintResolver(field)

    for regime in [Regime.STRICT, Regime.EXPANSIVE]:
        print(f"\n--- {regime.name} PROPAGATION & GRADIENTS ---")
        state = resolver.propagate(
            anchors={"absorption_identity": 1.0},
            target_node="justificative_displacement",
            depth=3,
            contextual_gain=0.4,
            local_decay=1.0,
            regime=regime,
        )

        print(f"Loss (L = -activation[target]): {state.loss:.4f}")

        print("\nNode Activation Values:")
        for key, value in sorted(state.active_nodes.items()):
            print(f"  {key:28} {value:.4f}")

        if plot:
            plot_relational_field(
                state,
                field,
                f"relational_field_analysis_{regime.value}.png",
                title=f"Relational Field Activation (Regime: {regime.name})",
            )
            if regime == Regime.STRICT:
                plot_relational_field(
                    state,
                    field,
                    "relational_field_analysis.png",
                    title="Relational Field Activation (STRICT)",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relational Field Simulation")
    parser.add_argument("--simulate", action="store_true", help="Run simulation")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    if args.simulate:
        run_simulation(plot=args.plot)
    else:
        # Default behavior for compatibility with previous run
        run_simulation(plot=False)
