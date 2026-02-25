import torch
import matplotlib.pyplot as plt
import numpy as np
from concept_routing import ConceptRoutingModel

def run_analysis():
    print("Running Concept Routing Analysis...")
    vocab_size = 100
    embed_dim = 64
    num_clusters = 6
    num_response_types = 4
    seq_len = 15

    model = ConceptRoutingModel(vocab_size, embed_dim, num_clusters, num_response_types)
    model.eval()

    tokens = torch.randint(0, vocab_size, (seq_len,))

    with torch.no_grad():
        output = model(tokens)
        cluster_probs = output["cluster_probs"].numpy()
        intent_probs = output["intent_probs"].numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Cluster Probs
    clusters = [f'Cluster {i}' for i in range(num_clusters)]
    ax1.bar(clusters, cluster_probs, color='skyblue')
    ax1.set_title('Concept Cluster Distribution')
    ax1.set_ylabel('Probability')
    ax1.set_ylim(0, 1)

    # Intent Probs
    intents = [f'Intent {i}' for i in range(num_response_types)]
    ax2.pie(intent_probs, labels=intents, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    ax2.set_title('Discursive Intent Inference')

    plt.tight_layout()
    plt.savefig("routing_analysis.png")
    print("Analysis complete. Visualization saved to routing_analysis.png")

if __name__ == "__main__":
    run_analysis()
