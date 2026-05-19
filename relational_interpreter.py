from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import networkx as nx


@dataclass
class Node:
    name: str
    description: str = ""
    children: List["Node"] = field(default_factory=list)
    relations: Dict[str, List["Node"]] = field(default_factory=dict)
    # Quadripartitioned determinacy profile
    formal_det: float = 0.0
    informal_det: float = 0.0
    formal_indet: float = 0.0
    informal_indet: float = 0.0

    def add_child(self, node: "Node") -> "Node":
        self.children.append(node)
        return self

    def relate(self, relation: str, node: "Node") -> "Node":
        self.relations.setdefault(relation, []).append(node)
        return self

    def traverse(self, depth: int = 0, seen: set = None) -> None:
        if seen is None:
            seen = set()
        if id(self) in seen:
            return
        seen.add(id(self))
        indent = "  " * depth
        print(f"{indent}[{self.name}]" + (f": {self.description}" if self.description else ""))
        for child in self.children:
            child.traverse(depth + 1, seen)
        for rel, nodes in self.relations.items():
            for node in nodes:
                print(f"{indent}  --{rel}--> [{node.name}]")

    def to_graph(self, G: Optional[nx.DiGraph] = None, seen: set = None) -> nx.DiGraph:
        if G is None:
            G = nx.DiGraph()
        if seen is None:
            seen = set()
        if id(self) in seen:
            return G
        seen.add(id(self))

        G.add_node(self.name, description=self.description,
                   formal_det=self.formal_det, informal_det=self.informal_det,
                   formal_indet=self.formal_indet, informal_indet=self.informal_indet)

        for child in self.children:
            G.add_edge(self.name, child.name, relation="child")
            child.to_graph(G, seen)

        for rel, nodes in self.relations.items():
            for node in nodes:
                G.add_edge(self.name, node.name, relation=rel)
                node.to_graph(G, seen)

        return G

    def get_role(self, G: nx.DiGraph) -> str:
        """
        Determines if a unit's role is constitutive or cosmetic.
        Constitutive: High formal/informal determinacy and high centrality.
        Cosmetic: High formal/informal indeterminacy or low centrality.
        """
        try:
            centrality = nx.degree_centrality(G)[self.name]
        except KeyError:
            centrality = 0.0

        determinacy = (self.formal_det + self.informal_det) / 2
        indeterminacy = (self.formal_indet + self.informal_indet) / 2

        # Weighted score: (Det - Indet) * Centrality
        score = (determinacy - indeterminacy) * (1 + centrality)

        return "constitutive" if score > 0.5 else "cosmetic"


def couple(o: Node, s: Node, c: Node) -> Node:
    g = Node(name="G", description="Derived relational output within synthetic field")
    for constraint in c.children:
        o.relate("constrained_by", constraint)
        s.relate("modulated_by", constraint)
    for lens in s.children:
        o.relate("projected_through", lens)
    for component in o.children + s.children + c.children:
        g.relate("induced_from", component)
    g.relate("via_coupling_of", o)
    g.relate("via_coupling_of", s)
    g.relate("via_coupling_of", c)
    return g


O = Node(name="O", description="Computational substrate")
O.add_child(Node("embeddings"))
O.add_child(Node("token_configurations"))
O.add_child(Node("state_invariants"))
O.add_child(Node("consistency_regions"))

S = Node(name="S", description="Interpretation regime")
S.add_child(Node("lexical_lens"))
S.add_child(Node("geometric_lens"))
S.add_child(Node("probabilistic_lens"))
S.add_child(Node("interpretive_mapping"))

C = Node(name="C", description="Coupling constraints")
C.add_child(Node("alignment_constraints"))
C.add_child(Node("span_relations"))
C.add_child(Node("threshold_interfaces"))
C.add_child(Node("cross_domain_bindings"))


if __name__ == "__main__":
    G = couple(O, S, C)
    print("\nSYNTHETIC RELATIONAL DOMAIN\n")
    G.traverse()
