import torch
import matplotlib.pyplot as plt
import networkx as nx
from relational_interpreter import Node, ONode, CNode, SNode, TriadicSystem, GNode, DIM, VOCAB_SIZE

def run_analysis():
    torch.manual_seed(42)

    # 1. Build a Triadic System with explicit determinacy
    # O (Substrate)
    O = ONode(DIM, formal_det=0.8, informal_det=0.7)

    # C (Coupling)
    C = CNode(DIM, formal_det=0.9, informal_indet=0.1)

    # S (Regime)
    S = SNode(DIM, formal_det=0.5, informal_indet=0.5)

    # Also set some children for more interesting topo
    O.children[0].formal_det = 0.9  # embeddings
    C.children[0].formal_det = 0.95 # alignment

    T1 = TriadicSystem(O, C, S, name="T1_Manifold", formal_det=0.9, informal_det=0.8)

    # 2. Build a second manifold for coupling
    T2 = TriadicSystem(ONode(DIM), CNode(DIM), SNode(DIM), name="T2_Manifold", formal_det=0.4, informal_indet=0.6)

    # 3. Create the Synthetic Relational Field (G)
    G_field = GNode(T1, T2, dim=DIM, name="G_Emergence", formal_det=0.95, informal_det=0.9)

    # 4. Functional Verification
    B, T = 1, 8
    ids1 = torch.randint(0, VOCAB_SIZE, (B, T))
    ids2 = torch.randint(0, VOCAB_SIZE, (B, T))

    with torch.no_grad():
        output = G_field(ids1, ids2)

    print(f"Emergent Field Output Shape: {tuple(output.shape)}")

    # 5. Topological Analysis
    G_topo = G_field.to_graph()

    # Calculate roles
    # We collect all sub-modules that were registered in the graph
    roles = {}
    node_objects = {}

    # Helper to find Node/Module by name from our constructed hierarchy
    def collect_nodes(module, m_dict):
        if hasattr(module, "node_name"):
            m_dict[module.node_name] = module
        # Node and its subclasses have a 'children' list attribute
        # nn.Module has a 'children' method
        children_attr = getattr(module, "children", None)
        if isinstance(children_attr, list):
            for child in children_attr:
                collect_nodes(child, m_dict)

        if hasattr(module, "left"):
            collect_nodes(module.left, m_dict)
        if hasattr(module, "right"):
            collect_nodes(module.right, m_dict)
        if hasattr(module, "O"): collect_nodes(module.O, m_dict)
        if hasattr(module, "C"): collect_nodes(module.C, m_dict)
        if hasattr(module, "S"): collect_nodes(module.S, m_dict)

    collect_nodes(G_field, node_objects)

    for node_name in G_topo.nodes():
        node_obj = node_objects.get(node_name)
        if node_obj and hasattr(node_obj, "get_role"):
            roles[node_name] = node_obj.get_role(G_topo)
        else:
            # Fallback for simple Nodes or missing objects
            # We can use the logic from Node.get_role if we have the attributes
            data = G_topo.nodes[node_name]
            f_det = data.get("formal_det", 0)
            i_det = data.get("informal_det", 0)
            f_indet = data.get("formal_indet", 0)
            i_indet = data.get("informal_indet", 0)

            try:
                centrality = nx.degree_centrality(G_topo)[node_name]
            except KeyError:
                centrality = 0.0

            det = (f_det + i_det) / 2
            indet = (f_indet + i_indet) / 2
            score = (det - indet) * (1 + centrality)
            roles[node_name] = "constitutive" if score > 0.5 else "cosmetic"

    # 6. Visualization
    plt.figure(figsize=(16, 10))
    pos = nx.kamada_kawai_layout(G_topo)

    color_map = ["lightgreen" if roles.get(node) == "constitutive" else "salmon" for node in G_topo.nodes()]

    nx.draw(G_topo, pos, with_labels=True, node_color=color_map, node_size=2000, font_size=8, font_weight="bold", arrows=True, alpha=0.8)

    edge_labels = nx.get_edge_attributes(G_topo, "relation")
    nx.draw_networkx_edge_labels(G_topo, pos, edge_labels=edge_labels, font_color="blue", font_size=7)

    plt.title("Synthetic Relational Field (G) Topology: Manifold Interaction & Determinacy Roles")
    plt.savefig("relational_analysis.png")
    print("Analysis complete. Saved visualization to relational_analysis.png")

    print("\nDeterminacy Report:")
    print(f"{'Node Name':<25} | {'Role':<15} | {'Det':<6} | {'Indet':<6}")
    print("-" * 60)
    for name in sorted(G_topo.nodes()):
        data = G_topo.nodes[name]
        det = (data.get("formal_det", 0) + data.get("informal_det", 0)) / 2
        indet = (data.get("formal_indet", 0) + data.get("informal_indet", 0)) / 2
        print(f"{name:<25} | {roles[name]:<15} | {det:<6.2f} | {indet:<6.2f}")

if __name__ == "__main__":
    run_analysis()
