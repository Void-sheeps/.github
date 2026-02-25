import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class ConceptRoutingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_clusters, num_response_types):
        super().__init__()

        # Embedding base
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Protótipos de cluster (ex: cálculo, filosofia, didática...)
        self.cluster_prototypes = nn.Parameter(
            torch.randn(num_clusters, embed_dim)
        )

        # Classificador de intenção discursiva
        self.intent_classifier = nn.Linear(embed_dim, num_response_types)

    def forward(self, tokens):
        """
        tokens: (seq_len)
        """

        # 1. Embedding médio do contexto
        x = self.embedding(tokens)                 # (seq_len, dim)
        context_vec = x.mean(dim=0)               # (dim)

        # 2. Similaridade com clusters conceituais
        context_norm = F.normalize(context_vec, dim=0)
        clusters_norm = F.normalize(self.cluster_prototypes, dim=1)

        cluster_scores = torch.matmul(clusters_norm, context_norm)
        cluster_probs = F.softmax(cluster_scores, dim=0)

        # 3. Inferência de tipo de resposta
        intent_logits = self.intent_classifier(context_vec)
        intent_probs = F.softmax(intent_logits, dim=0)

        return {
            "cluster_probs": cluster_probs,
            "intent_probs": intent_probs
        }

if __name__ == "__main__":
    if "--simulate" in sys.argv:
        print("Starting ConceptRoutingModel simulation...")
        vocab_size = 100
        embed_dim = 64
        num_clusters = 5
        num_response_types = 3
        seq_len = 12

        model = ConceptRoutingModel(vocab_size, embed_dim, num_clusters, num_response_types)
        tokens = torch.randint(0, vocab_size, (seq_len,))

        output = model(tokens)
        print("Cluster Probs:", output["cluster_probs"])
        print("Intent Probs:", output["intent_probs"])
        print("Simulation complete.")
