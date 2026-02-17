#!/usr/bin/env python3
"""
Advanced Narrative Immersion Field Analysis: Dynamical Extensions
===============================================================
Extends the static field model into a dynamical system with:
1. Hysteresis (The "Trust Regaining" friction)
2. Cusp Catastrophe Visualization (The "Shark Jump" topology)
3. Narrative Entropy (Information density metric)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DynamicalNarrativeModel:
    def __init__(self):
        # Decay rates and recovery rates
        self.alpha_decay = 0.3    # How fast Suspension dies on bad coherence
        self.beta_recovery = 0.05 # How slow Suspension builds (Trust is hard to earn)
        self.memory_s = 0.8       # Initial suspension

    def step_dynamics(self, C_target, S_current, dt=0.1):
        """
        Calculates dS/dt based on current Coherence.
        Model: dS/dt = Recovery_Rate * (1-S) if C is high
                       - Decay_Rate * S      if C is low
        This creates a 'Lag' or Hysteresis effect.
        """
        # Sigmoid activation for coherence threshold
        # If Coherence > 0.6, we recover. If < 0.6, we decay.
        coherence_pressure = (1 / (1 + np.exp(-10 * (C_target - 0.6))))

        # Differential equation for Suspension
        # If pressure is high: Grow towards 1.0 (slowly)
        # If pressure is low: Decay towards 0.0 (quickly)
        dS_growth = self.beta_recovery * (1.0 - S_current) * coherence_pressure
        dS_decay = -self.alpha_decay * S_current * (1.0 - coherence_pressure)

        dS = dS_growth + dS_decay

        return np.clip(S_current + dS * dt, 0, 1)

    def calculate_entropy(self, C, E, A):
        """
        Calculates Narrative Entropy (H).
        High Entropy = Confusion/Chaos (Low C, High A).
        Low Entropy = Boredom/Predictability (High C, Low E).
        Sweet spot = 'Complex Order' (Edge of Chaos).
        """
        # Normalize inputs to probability-like distributions for Shannon approximation
        # This is a heuristic metric for 'Narrative Complexity'
        p_c = np.clip(C, 0.01, 0.99)
        p_e = np.clip(E, 0.01, 0.99)

        # Entropy contribution from Coherence (inverted, low C is high entropy)
        h_c = -(p_c * np.log2(p_c) + (1-p_c) * np.log2(1-p_c))

        # Total System Energy ~ Entropy * Amplitude
        return h_c * A

def visualize_hysteresis_loop():
    """
    Demonstrates that regaining immersion is harder than losing it.
    The 'Fool me once' effect.
    """
    print("Generating Hysteresis Loop simulation...")
    model = DynamicalNarrativeModel()

    # Create a coherence cycle: High -> Low -> High
    t = np.linspace(0, 20, 200)
    # Coherence drops from 1.0 to 0.0 then goes back to 1.0
    C_input = 0.5 + 0.5 * np.cos(t * 0.5)

    S_output = []
    current_S = 0.9 # Start with high trust

    for c in C_input:
        # Run multiple sub-steps to simulate time passing for each input point
        for _ in range(5):
            current_S = model.step_dynamics(c, current_S, dt=0.1)
        S_output.append(current_S)

    S_output = np.array(S_output)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Time Series
    ax1.plot(t, C_input, 'b--', label='Coherence Input (Script Quality)')
    ax1.plot(t, S_output, 'r-', linewidth=3, label='Suspension of Disbelief (Audience State)')
    ax1.set_title('Temporal Lag: The "Trust" Delay')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.annotate('Rapid Collapse', xy=(6.5, 0.4), xytext=(8, 0.6),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    ax1.annotate('Slow Recovery', xy=(16, 0.6), xytext=(12, 0.8),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # Phase Portrait (Hysteresis Loop)
    ax2.plot(C_input, S_output, 'g-', linewidth=2)

    # Add arrows to show direction
    mid_idx = len(C_input)//2
    ax2.arrow(C_input[50], S_output[50], C_input[51]-C_input[50], S_output[51]-S_output[50],
              head_width=0.03, color='g')
    ax2.arrow(C_input[150], S_output[150], C_input[151]-C_input[150], S_output[151]-S_output[150],
              head_width=0.03, color='g')

    ax2.set_title('Hysteresis Loop: The Cost of Breaking Immersion')
    ax2.set_xlabel('Coherence (Input)')
    ax2.set_ylabel('Suspension of Disbelief (Result)')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, 0.2, "Broken State", ha='center', color='red')
    ax2.text(0.5, 0.8, "Immersed State", ha='center', color='green')

    plt.tight_layout()
    plt.savefig('narrative_hysteresis.png')
    print("✓ Hysteresis Loop saved to narrative_hysteresis.png")
    plt.close()

def visualize_catastrophe_manifold():
    """
    Visualizes the 'Narrative Cusp'.
    Shows how High Emotion + Dropping Coherence leads to a catastrophic
    drop in immersion (The 'Betrayal' or 'Shark Jump'), whereas
    Low Emotion + Dropping Coherence just leads to boredom (smooth decline).
    """
    print("Generating Cusp Catastrophe Manifold...")

    # Grid
    n = 50
    coherence = np.linspace(0, 1, n)
    emotion = np.linspace(0, 1, n)
    C, E = np.meshgrid(coherence, emotion)

    # Manifold Logic
    Immersion = C * E
    penalty = (E ** 2) * (np.exp(-10 * (C - 0.4)))
    Z = Immersion - penalty * 0.5
    Z = np.clip(Z, 0, 1)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(C, E, Z, cmap='plasma', edgecolor='none', alpha=0.9)

    ax.set_xlabel('Coherence (Logic)')
    ax.set_ylabel('Emotional Resonance')
    ax.set_zlabel('Resulting Immersion')
    ax.set_title('The Narrative Cusp: High Stakes make Plot Holes Fatal')

    ax.text(0.2, 0.9, 0.1, "The Uncanny Valley /\nShark Jump", color='black', fontsize=10, weight='bold')
    ax.text(0.9, 0.9, 0.9, "Deep Flow", color='white', fontsize=10)

    fig.colorbar(surf, shrink=0.5, aspect=5, label='Immersion Depth')
    plt.savefig('narrative_catastrophe.png')
    print("✓ Catastrophe Manifold saved to narrative_catastrophe.png")
    plt.close()

def narrative_entropy_map():
    """
    Maps the 'Texture' of the narrative.
    High Entropy = Chaotic/Confusing
    Low Entropy = Boring
    Mid Entropy = Engaging
    """
    print("Generating Entropy Map...")
    model = DynamicalNarrativeModel()

    c_vals = np.linspace(0.01, 0.99, 100)
    a_vals = np.linspace(0.01, 0.99, 100) # Attention
    C_grid, A_grid = np.meshgrid(c_vals, a_vals)

    # Fix Emotion
    E_fixed = 0.6

    Entropy = model.calculate_entropy(C_grid, E_fixed, A_grid)

    plt.figure(figsize=(10, 8))
    cp = plt.contourf(C_grid, A_grid, Entropy, levels=20, cmap='inferno')
    plt.colorbar(cp, label='Narrative Entropy (Cognitive Load)')

    plt.xlabel('Coherence (structure)')
    plt.ylabel('Attentional Capture (Spectacle)')
    plt.title('Narrative Entropy: The "Michael Bay" vs "Art House" Spectrum')

    plt.text(0.1, 0.9, 'Pure Chaos\n(Michael Bay)', color='white', ha='center')
    plt.text(0.9, 0.1, 'Dry Lecture', color='white', ha='center')
    plt.text(0.5, 0.5, 'Complex\nEngagement', color='white', ha='center', weight='bold')

    plt.savefig('narrative_entropy.png')
    print("✓ Entropy Map saved to narrative_entropy.png")
    plt.close()

if __name__ == "__main__":
    print("="*60)
    print("DYNAMICAL NARRATIVE IMMERSION FIELD ANALYSIS")
    print("="*60)

    visualize_hysteresis_loop()
    visualize_catastrophe_manifold()
    narrative_entropy_map()

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
