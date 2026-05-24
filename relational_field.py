from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np


class Relation(Enum):
    LOCAL = "local"
    TOPOLOGICAL = "topological"
    CONTEXTUAL = "contextual"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    JUSTIFICATIVE = "justificative"


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


@dataclass
class FieldState:
    active_nodes: Dict[str, float] = field(default_factory=dict)
    active_edges: List[Edge] = field(default_factory=list)
    contextual_pressure: float = 1.0
    local_rigidity: float = 1.0

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
        return [
            edge
            for edge in self.edges
            if edge.source == node_id
        ]


class ConstraintResolver:
    def __init__(self, field: SemanticField) -> None:
        self.field = field

    def propagate(
        self,
        anchors: Dict[str, float],
        depth: int = 3,
        contextual_gain: float = 1.0,
        local_decay: float = 0.5,
    ) -> FieldState:
        state = FieldState(
            contextual_pressure=contextual_gain,
            local_rigidity=local_decay,
        )

        for node_id, magnitude in anchors.items():
            state.inject(node_id, magnitude)

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
                    propagated = self._resolve_energy(
                        edge=edge,
                        energy=energy,
                        contextual_gain=contextual_gain,
                        local_decay=local_decay,
                    )

                    if propagated <= 0:
                        continue

                    state.inject(edge.target, propagated)
                    state.active_edges.append(edge)
                    next_frontier.append((edge.target, propagated))

            frontier = next_frontier

        state.normalize()
        return state

    @staticmethod
    def _resolve_energy(
        edge: Edge,
        energy: float,
        contextual_gain: float,
        local_decay: float,
    ) -> float:
        continuity_relations = {
            Relation.TOPOLOGICAL,
            Relation.CONTEXTUAL,
            Relation.ANALOGICAL,
        }

        local_relations = {
            Relation.LOCAL,
            Relation.CAUSAL,
            Relation.JUSTIFICATIVE,
        }

        if edge.relation in continuity_relations:
            modifier = contextual_gain
        elif edge.relation in local_relations:
            modifier = local_decay
        else:
            modifier = 1.0

        return energy * edge.weight * modifier


def run_analysis(simulate: bool = False, plot: bool = False) -> None:
    if not simulate and not plot:
        return

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

    resolver = ConstraintResolver(field)

    strict_state = resolver.propagate(
        anchors={"absorption_identity": 1.0},
        depth=3,
        contextual_gain=0.4,
        local_decay=1.0,
    )

    expansive_state = resolver.propagate(
        anchors={"absorption_identity": 1.0},
        depth=3,
        contextual_gain=1.3,
        local_decay=0.6,
    )

    if simulate:
        print("--- Relational Field Simulation ---")
        print("STRICT")
        for key, value in sorted(strict_state.active_nodes.items()):
            print(f"{key:28} {value:.4f}")

        print()
        print("EXPANSIVE")
        for key, value in sorted(expansive_state.active_nodes.items()):
            print(f"{key:28} {value:.4f}")

    if plot:
        nodes = sorted(set(strict_state.active_nodes.keys()) | set(expansive_state.active_nodes.keys()))
        strict_vals = [strict_state.active_nodes.get(n, 0.0) for n in nodes]
        expansive_vals = [expansive_state.active_nodes.get(n, 0.0) for n in nodes]

        x = np.arange(len(nodes))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 7))
        rects1 = ax.bar(x - width/2, strict_vals, width, label='Strict', color='steelblue')
        rects2 = ax.bar(x + width/2, expansive_vals, width, label='Expansive', color='salmon')

        ax.set_ylabel('Activation Magnitude')
        ax.set_title('Relational Field State Comparison: Strict vs Expansive')
        ax.set_xticks(x)
        ax.set_xticklabels(nodes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig("relational_field_analysis.png")
        print("\nAnalysis complete. Visualization saved to relational_field_analysis.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relational Field Constraint Resolver Simulation")
    parser.add_argument("--simulate", action="store_true", help="Run the engine simulation")
    parser.add_argument("--plot", action="store_true", help="Generate analysis plots")
    args = parser.parse_args()

    if not any(vars(args).values()):
        run_analysis(simulate=True)
    else:
        run_analysis(simulate=args.simulate, plot=args.plot)
