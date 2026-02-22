#!/usr/bin/env python3
"""
merge_module_temporal.py - Merge module with timing and retro-convergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
from typing import List, Tuple, Optional
from antinomy_resolver import NeuralAntinomy, Atom

class MergeModuleTemporal(nn.Module):
    def __init__(self, similarity_threshold: float = 0.5, retro_convergence: float = 0.2, antinomy_resolver: Optional[NeuralAntinomy] = None):
        """
        Args:
            similarity_threshold: Minimum similarity to merge tokens.
            retro_convergence: Weight of previous merge embeddings influencing current step.
            antinomy_resolver: Optional NeuralAntinomy module to detect and handle conflicts.
        """
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.retro_convergence = retro_convergence
        self.antinomy_resolver = antinomy_resolver
        self.previous_merges = {}

    @staticmethod
    def similarity_metric(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

    @staticmethod
    def fuse_embeddings(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        return torch.clamp(vec1 + vec2, max=1.0)

    def forward(self, tokens: List[str], embeddings: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        input_start_time = time.time()

        current_tokens = tokens.copy()
        current_embs_list = [embeddings[i].clone() for i in range(embeddings.size(0))]

        step = 0
        while True:
            n = len(current_tokens)
            if n <= 1:
                break

            current_embs_tensor = torch.stack(current_embs_list)

            # Phase 0: Antinomy Resolution at each step
            inconsistent_indices = set()
            if self.antinomy_resolver is not None:
                atoms = [Atom(t) for t in current_tokens]
                current_embs_tensor, conflict_scores, inconsistent_atoms = self.antinomy_resolver(current_embs_tensor, atoms)
                inconsistent_names = {a.name for a in inconsistent_atoms}
                inconsistent_indices = {i for i, t in enumerate(current_tokens) if t in inconsistent_names}
                if inconsistent_indices:
                    print(f"  [Antinomy] Inconsistency detected in tokens: {[current_tokens[i] for i in inconsistent_indices]}")

            current_embs = [current_embs_tensor[i] for i in range(current_embs_tensor.size(0))]
            if n <= 1:
                break

            merged_indices = set()
            new_tokens = []
            new_embs = []
            merge_occurred = False

            for i in range(n):
                if i in merged_indices:
                    continue
                best_j = None
                best_score = self.similarity_threshold
                for j in range(i + 1, n):
                    if j in merged_indices:
                        continue

                    # Heuristic: Block merging if either token is inconsistent
                    if i in inconsistent_indices or j in inconsistent_indices:
                        continue

                    score = self.similarity_metric(current_embs[i], current_embs[j])
                    if score >= best_score:
                        best_score = score
                        best_j = j

                if best_j is not None:
                    merged_emb = self.fuse_embeddings(current_embs[i], current_embs[best_j])

                    # Retro-convergence: adicionar influência de merges anteriores
                    prev_key = f"{current_tokens[i]}|{current_tokens[best_j]}"
                    if prev_key in self.previous_merges:
                        merged_emb = (1 - self.retro_convergence) * merged_emb + \
                                     self.retro_convergence * self.previous_merges[prev_key]

                    merged_token = f"{current_tokens[i]}|{current_tokens[best_j]}"
                    new_tokens.append(merged_token)
                    new_embs.append(merged_emb)
                    self.previous_merges[merged_token] = merged_emb
                    merged_indices.add(i)
                    merged_indices.add(best_j)
                    merge_occurred = True
                else:
                    new_tokens.append(current_tokens[i])
                    new_embs.append(current_embs[i])
                    merged_indices.add(i)

            current_tokens = new_tokens
            current_embs_list = new_embs
            step += 1

            if not merge_occurred:
                break

        output_time = time.time()
        print(f"[MergeModuleTemporal] Steps: {step}, Input time: {input_start_time:.4f}s, "
              f"Output time: {output_time:.4f}s, Duration: {output_time-input_start_time:.4f}s")

        merged_embeddings = torch.stack(current_embs_list) if current_embs_list else torch.empty(0)
        return current_tokens, merged_embeddings

def run_simulation():
    print("--- Running MergeModuleTemporal Simulation ---")
    tokens = ["cat", "bat", "dog", "dot"]
    embeddings = torch.tensor([
        [0.9, 0.1, 0.0],
        [0.85, 0.15, 0.0],
        [0.1, 0.9, 0.0],
        [0.15, 0.85, 0.0]
    ])

    # Threshold 0.7 works now with cosine similarity
    merge_module = MergeModuleTemporal(similarity_threshold=0.7, retro_convergence=0.3)

    # 1st run
    print("\nInitial tokens:", tokens)
    merged_tokens1, merged_embeddings1 = merge_module(tokens, embeddings)
    print("Merged Tokens 1:", merged_tokens1)

    # 2nd run: Simular um novo input com os mesmos tokens originais
    # A retro-convergência deve atuar quando os mesmos tokens forem mesclados novamente
    new_embeddings = torch.tensor([
        [0.95, 0.05, 0.0],
        [0.8, 0.2, 0.0],
        [0.05, 0.95, 0.0],
        [0.2, 0.8, 0.0]
    ])
    print("\n2nd run with same original tokens (retro-convergence active):")
    merged_tokens2, merged_embeddings2 = merge_module(tokens, new_embeddings)
    print("Merged Tokens 2:", merged_tokens2)
    print("\nSimulation Complete.")

# ==============================
# CLI Entry Point
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Module Temporal Simulation")
    parser.add_argument("--simulate", action="store_true", help="Run the merge simulation")
    args = parser.parse_args()

    if args.simulate or not any(vars(args).values()):
        run_simulation()
