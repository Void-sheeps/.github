#!/usr/bin/env python3
"""
Proof-of-Thought Engine
=======================

A hierarchical reasoning system simulating AI cognition:
- Each thought is a node with content, vectorized meaning, and metrics.
- Recursive aggregation propagates coherence, suspension, and critical distance.
- Fatigue and dissimulation are incorporated to simulate cognitive limits.

Modules integrated from conversation:
- NarrativeImmersionField (C, E, S, A metrics)
- Hierarchical Node vectorization (tanh aggregation)
- Fatigue / desensitization dynamics
- Snapshot system for reasoning trace
"""

import numpy as np
import argparse

# -----------------------------
# Meaning Vector Node
# -----------------------------
DIM = 8

class ThoughtNode:
    def __init__(self, label, content, children=None):
        self.label = label
        self.content = content
        self.children = children or []
        self.vector = self.text_to_vector(content)
        self.C = 1.0  # Coherence
        self.S = 1.0  # Suspension of disbelief
        self.CD = 0.0 # Critical distance
        self.fatigue = 0.0  # Cognitive fatigue

    def text_to_vector(self, text):
        """Deterministic pseudo-vector for content"""
        h = hash(text)
        np.random.seed(h % (2**32))
        return np.random.uniform(-1,1,DIM)

    def aggregate(self):
        """Aggregate meaning recursively (tanh activation)"""
        if not self.children:
            return self.vector
        child_vecs = np.array([child.aggregate() for child in self.children])
        agg = np.tanh(np.mean(np.vstack([self.vector, child_vecs]), axis=0))
        self.vector = agg
        return agg

    def update_metrics(self):
        """Update coherence, suspension, critical distance recursively"""
        if not self.children:
            self.C = 1.0
            self.S = 1.0
        else:
            child_vectors = np.array([child.vector for child in self.children])
            mean_child = np.mean(child_vectors, axis=0)
            sim = np.dot(self.vector, mean_child)/(np.linalg.norm(self.vector)*np.linalg.norm(mean_child)+1e-8)
            self.C = np.clip(sim, 0, 1)
            child_Cs = np.array([child.C for child in self.children])
            self.S = np.prod(child_Cs)**0.5  # geometric mean
        self.CD = self.C * (1 - self.S)
        # propagate recursively
        for child in self.children:
            child.update_metrics()

    def apply_fatigue(self, dt=0.1):
        """Fatigue builds with cognitive load (high amplitude / misalignment)"""
        if self.children:
            avg_child_amp = np.mean([np.linalg.norm(child.vector) for child in self.children])
            dF = 0.05 * avg_child_amp - 0.01 * (1 - avg_child_amp)
            self.fatigue = np.clip(self.fatigue + dF*dt, 0, 1)
        for child in self.children:
            child.apply_fatigue(dt)

    def snapshot(self, depth=0):
        """Print recursive proof-of-thought trace"""
        print("  "*depth + f"{self.label} | C:{self.C:.2f} S:{self.S:.2f} CD:{self.CD:.2f} F:{self.fatigue:.2f}")
        for child in self.children:
            child.snapshot(depth+1)

# -----------------------------
# Narrative Immersion Field Module
# -----------------------------
class NarrativeImmersionField:
    def forward(self, C, E, S, A):
        C, E, S, A = map(lambda x: np.clip(x, 0, 1), (C,E,S,A))
        I_depth = C * E * S
        N_int = C * E
        CD = N_int * (1 - S)
        return {"immersion_depth": I_depth, "narrative_integrity": N_int, "critical_distance": CD}

# -----------------------------
# Proof-of-Thought Simulation
# -----------------------------
def simulate_proof_of_thought():
    print("="*60)
    print("PROOF-OF-THOUGHT SIMULATION")
    print("="*60)

    # Build hierarchical reasoning tree
    root = ThoughtNode("Root Thought", "Should AI dissimulate in reasoning?")

    step1 = ThoughtNode("Step 1", "Assess narrative coherence.")
    step1.children.append(ThoughtNode("Substep 1.1", "Evaluate internal consistency of content."))
    step1.children.append(ThoughtNode("Substep 1.2", "Compare with known truths."))

    step2 = ThoughtNode("Step 2", "Assess audience engagement.")
    step2.children.append(ThoughtNode("Substep 2.1", "Compute immersion depth."))
    step2.children.append(ThoughtNode("Substep 2.2", "Check for plot holes or contradictions."))

    step3 = ThoughtNode("Step 3", "Determine cognitive fatigue.")
    step3.children.append(ThoughtNode("Substep 3.1", "High misalignment increases fatigue."))

    root.children = [step1, step2, step3]

    # Introduce a "misalignment" (plot hole / contradiction) in Substep 2.2
    root.children[1].children[1].vector *= 0.2

    # Aggregate vectors, update metrics, and simulate fatigue
    root.aggregate()
    root.update_metrics()
    root.apply_fatigue()

    # Snapshot proof-of-thought
    root.snapshot()

    # Integrate with NarrativeImmersionField
    field = NarrativeImmersionField()
    metrics = field.forward(root.C, 0.8, root.S, 0.9)  # Example E=0.8, A=0.9
    print("\nNarrative Immersion Metrics:")
    print(f"Immersion Depth: {metrics['immersion_depth']:.3f}")
    print(f"Narrative Integrity: {metrics['narrative_integrity']:.3f}")
    print(f"Critical Distance: {metrics['critical_distance']:.3f}")

# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proof-of-Thought Engine Simulation")
    parser.add_argument("--simulate", action="store_true", help="Run the engine simulation")
    args = parser.parse_args()

    if args.simulate or not any(vars(args).values()):
        simulate_proof_of_thought()
