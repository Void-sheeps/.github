import matplotlib.pyplot as plt
import networkx as nx
from relational_interpreter import Node, couple

def run_analysis():
    # Define Substrate (O) with varied determinacy
    O = Node(name="O", description="Substrate", formal_det=0.8, informal_det=0.7)
    O.add_child(Node("embeddings", formal_det=0.9, informal_det=0.8))
    O.add_child(Node("token_configurations", formal_det=0.6, informal_indet=0.4))
    O.add_child(Node("state_invariants", formal_det=1.0, informal_det=1.0))

    # Define Regime (S) with varied determinacy
    S = Node(name="S", description="Regime", formal_det=0.5, informal_indet=0.5)
    S.add_child(Node("lexical_lens", formal_det=0.7, informal_det=0.6))
    S.add_child(Node("geometric_lens", formal_det=0.8, informal_det=0.7))

    # Define Constraints (C) - mostly constitutive
    C = Node(name="C", description="Constraints", formal_det=0.9, formal_indet=0.1)
    C.add_child(Node("alignment_constraints", formal_det=0.95))
    C.add_child(Node("threshold_interfaces", informal_det=0.85))

    # Couple them
    G_node = couple(O, S, C)

    # Export to networkx
    G = G_node.to_graph()

    # Calculate roles for each node in the graph
    # We need a mapping from name to Node object for easier access,
    # but since Node.to_graph only uses names, we'll re-instantiate or find them.
    # For simplicity in this analysis, we'll assume the names are unique and match.
    nodes_map = {
        "O": O, "embeddings": O.children[0], "token_configurations": O.children[1], "state_invariants": O.children[2],
        "S": S, "lexical_lens": S.children[0], "geometric_lens": S.children[1],
        "C": C, "alignment_constraints": C.children[0], "threshold_interfaces": C.children[1],
        "G": G_node
    }

    roles = {}
    for name, node in nodes_map.items():
        roles[name] = node.get_role(G)

    # Visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)

    # Color based on roles
    color_map = ["lightgreen" if roles.get(node, "cosmetic") == "constitutive" else "salmon" for node in G.nodes()]

    nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=3000, font_size=10, font_weight="bold", arrows=True)

    edge_labels = nx.get_edge_attributes(G, "relation")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="blue")

    plt.title("Relational Interpreter Topology: Constitutive (Green) vs Cosmetic (Red) Roles")
    plt.savefig("relational_analysis.png")
    print("Analysis complete. Saved visualization to relational_analysis.png")

    print("\nNode Roles:")
    for name, role in roles.items():
        node = nodes_map[name]
        det = (node.formal_det + node.informal_det) / 2
        indet = (node.formal_indet + node.informal_indet) / 2
        print(f" - {name:25} | Role: {role:15} | Det: {det:.2f} | Indet: {indet:.2f}")

if __name__ == "__main__":
    run_analysis()
