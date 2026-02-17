#!/usr/bin/env python3
"""
Advanced Narrative Immersion Field Analysis
===========================================

Extensions to the base model including:
- Phase space visualization
- Bifurcation analysis
- Multi-agent simulation
- Temporal dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

class NarrativeImmersionField:
    """Base model"""
    def __init__(self):
        self.threshold_low = 0.2
        self.threshold_mid = 0.4
        self.threshold_high = 0.7
        self.threshold_very_high = 0.8

    def forward(self, C, E, S, A):
        C = np.clip(C, 0, 1)
        E = np.clip(E, 0, 1)
        S = np.clip(S, 0, 1)
        A = np.clip(A, 0, 1)

        immersion_depth = C * E * S
        narrative_integrity = C * E
        critical_distance = narrative_integrity * (1 - S)

        return {
            "immersion_depth": immersion_depth,
            "narrative_integrity": narrative_integrity,
            "critical_distance": critical_distance,
            "vars": (C, E, S, A)
        }

    def classify_state(self, C, E, S, A):
        if C < 0.2 and E < 0.2:
            return "static_noise"
        if C > 0.7 and E > 0.7 and S > 0.8:
            return "deep_flow_immersion"
        if C > 0.8 and E < 0.2 and A > 0.8:
            return "hollow_spectacle"
        if C > 0.7 and E < 0.3:
            return "dry_exposition"
        if C < 0.4 and E > 0.7 and S > 0.5:
            return "dream_logic"
        if C > 0.8 and E > 0.6 and S < 0.3:
            return "uncanny_valley_risk"
        return "liminal_state"

def visualize_phase_space():
    """
    Visualize the C-E phase space with state regions
    """
    print("Generating phase space visualization...")

    model = NarrativeImmersionField()

    # Create grid
    c_vals = np.linspace(0, 1, 100)
    e_vals = np.linspace(0, 1, 100)
    C_grid, E_grid = np.meshgrid(c_vals, e_vals)

    # Fixed values for S and A
    S_fixed = 0.85
    A_fixed = 0.5

    # Compute immersion depth across the grid
    immersion_grid = C_grid * E_grid * S_fixed

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Immersion Depth Heatmap
    ax1 = axes[0, 0]
    im1 = ax1.contourf(C_grid, E_grid, immersion_grid, levels=20, cmap='viridis')
    ax1.set_xlabel('Coherence (C)')
    ax1.set_ylabel('Emotional Resonance (E)')
    ax1.set_title('Immersion Depth (S=0.85, A=0.5)')
    plt.colorbar(im1, ax=ax1, label='Immersion Depth')

    # Add state boundaries
    ax1.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Static Noise threshold')
    ax1.axvline(x=0.2, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Deep Flow threshold')
    ax1.axvline(x=0.7, color='green', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=8)

    # Plot 2: Critical Distance
    critical_distance_grid = (C_grid * E_grid) * (1 - S_fixed)
    ax2 = axes[0, 1]
    im2 = ax2.contourf(C_grid, E_grid, critical_distance_grid, levels=20, cmap='coolwarm')
    ax2.set_xlabel('Coherence (C)')
    ax2.set_ylabel('Emotional Resonance (E)')
    ax2.set_title('Critical Distance (Analytical Mode)')
    plt.colorbar(im2, ax=ax2, label='Critical Distance')

    # Plot 3: State Classification Map
    ax3 = axes[1, 0]
    state_grid = np.zeros_like(C_grid)

    state_colors = {
        'static_noise': 0,
        'liminal_state': 1,
        'dream_logic': 2,
        'dry_exposition': 3,
        'hollow_spectacle': 4,
        'uncanny_valley_risk': 5,
        'deep_flow_immersion': 6
    }

    for i in range(len(c_vals)):
        for j in range(len(e_vals)):
            state = model.classify_state(c_vals[i], e_vals[j], S_fixed, A_fixed)
            state_grid[j, i] = state_colors[state]

    im3 = ax3.imshow(state_grid, extent=[0, 1, 0, 1], origin='lower',
                     cmap='tab10', aspect='auto', interpolation='nearest')
    ax3.set_xlabel('Coherence (C)')
    ax3.set_ylabel('Emotional Resonance (E)')
    ax3.set_title('Narrative State Classification')

    # Create legend
    legend_elements = [
        mpatches.Patch(color=plt.cm.tab10(state_colors[state]/10), label=state.replace('_', ' ').title())
        for state in state_colors.keys()
    ]
    ax3.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    # Plot 4: Immersion vs Critical Distance
    ax4 = axes[1, 1]

    # Sample trajectories
    trajectories = [
        ("Perfect Story", [(0.9, 0.9), (0.95, 0.95)]),
        ("Plot Hole Event", [(0.9, 0.9), (0.1, 0.9)]),
        ("Boring Lecture", [(0.9, 0.2), (0.95, 0.15)]),
    ]

    for name, points in trajectories:
        c_p = [p[0] for p in points]
        e_p = [p[1] for p in points]

        immersion = [c * e * S_fixed for c, e in points]
        critical_dist = [c * e * (1 - S_fixed) for c, e in points]

        ax4.plot(immersion, critical_dist, 'o-', label=name, markersize=8)
        ax4.annotate('Start', (immersion[0], critical_dist[0]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Add diagonal line (immersion = critical distance)
    max_val = 1.0
    ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Break-even line')

    ax4.set_xlabel('Immersion Depth')
    ax4.set_ylabel('Critical Distance')
    ax4.set_title('Immersion vs Analytical Viewing')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('narrative_phase_space.png', dpi=150, bbox_inches='tight')
    print("✓ Phase space visualization saved to narrative_phase_space.png")
    plt.close()

def temporal_dynamics_simulation():
    """
    Simulate a narrative arc over time
    """
    print("\nSimulating temporal narrative dynamics...")

    model = NarrativeImmersionField()

    # Define a story arc
    time_steps = 100
    t = np.linspace(0, 1, time_steps)

    # Story arc: Setup -> Rising tension -> Climax -> Plot hole -> Recovery attempt
    C_arc = np.concatenate([
        0.3 + 0.5 * t[:30],  # Rising coherence
        0.8 + 0.1 * np.sin(10 * t[30:50]),  # Stable high
        0.9 * np.ones(10),  # Peak
        0.1 * np.ones(10),  # Plot hole!
        0.1 + 0.3 * (t[70:] - t[70]),  # Attempted recovery (fixed index relative to t)
    ])

    E_arc = np.concatenate([
        0.4 + 0.4 * t[:30],  # Rising emotion
        0.8 + 0.15 * np.sin(8 * t[30:50] + np.pi/4),  # Emotional waves
        0.95 * np.ones(10),  # Peak emotion
        0.85 * np.ones(10),  # Emotion persists briefly
        0.85 - 0.3 * (t[70:] - t[70]),  # Emotional fatigue
    ])

    # Adjust lengths if they don't match time_steps due to slicing/concatenation
    if len(C_arc) > time_steps: C_arc = C_arc[:time_steps]
    if len(E_arc) > time_steps: E_arc = E_arc[:time_steps]

    # S reacts to C
    S_arc = 0.9 * (C_arc ** 0.5)  # Suspension tracks with coherence

    A_arc = 0.7 + 0.2 * np.sin(4 * t)  # Attention fluctuates

    # Compute metrics over time
    immersion = np.zeros(time_steps)
    integrity = np.zeros(time_steps)
    critical_dist = np.zeros(time_steps)
    states = []

    for i in range(time_steps):
        result = model.forward(C_arc[i], E_arc[i], S_arc[i], A_arc[i])
        immersion[i] = result['immersion_depth']
        integrity[i] = result['narrative_integrity']
        critical_dist[i] = result['critical_distance']
        states.append(model.classify_state(C_arc[i], E_arc[i], S_arc[i], A_arc[i]))

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Variables over time
    ax1 = axes[0]
    ax1.plot(t, C_arc, label='Coherence (C)', linewidth=2)
    ax1.plot(t, E_arc, label='Emotional Resonance (E)', linewidth=2)
    ax1.plot(t, S_arc, label='Suspension of Disbelief (S)', linewidth=2)
    ax1.plot(t, A_arc, label='Attentional Capture (A)', linewidth=2, alpha=0.6)
    ax1.axvline(x=0.6, color='red', linestyle='--', alpha=0.5, label='Plot Hole Event')
    ax1.set_ylabel('Value')
    ax1.set_title('Narrative Variables Over Time')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Plot 2: Derived metrics
    ax2 = axes[1]
    ax2.plot(t, immersion, label='Immersion Depth', linewidth=2, color='green')
    ax2.plot(t, integrity, label='Narrative Integrity', linewidth=2, color='blue')
    ax2.plot(t, critical_dist, label='Critical Distance', linewidth=2, color='orange')
    ax2.axvline(x=0.6, color='red', linestyle='--', alpha=0.5)
    ax2.fill_between(t, immersion, critical_dist, where=(immersion > critical_dist),
                     alpha=0.2, color='green', label='Immersed')
    ax2.fill_between(t, immersion, critical_dist, where=(immersion <= critical_dist),
                     alpha=0.2, color='red', label='Critical')
    ax2.set_ylabel('Metric Value')
    ax2.set_title('Derived Narrative Metrics')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: State timeline
    ax3 = axes[2]

    # Map states to colors
    state_color_map = {
        'static_noise': 0,
        'liminal_state': 1,
        'dream_logic': 2,
        'dry_exposition': 3,
        'hollow_spectacle': 4,
        'uncanny_valley_risk': 5,
        'deep_flow_immersion': 6
    }

    state_values = [state_color_map[s] for s in states]

    # Create colored bars
    for i in range(len(t)-1):
        color = plt.cm.tab10(state_values[i] / 10)
        ax3.barh(0, t[i+1] - t[i], left=t[i], height=0.5, color=color, edgecolor='none')

    ax3.set_xlim([0, 1])
    ax3.set_ylim([-0.5, 0.5])
    ax3.set_xlabel('Normalized Time')
    ax3.set_title('Narrative State Over Time')
    ax3.set_yticks([])

    # Add annotations
    ax3.annotate('Setup', (0.15, 0), fontsize=10, ha='center')
    ax3.annotate('Rising Action', (0.4, 0), fontsize=10, ha='center')
    ax3.annotate('Climax', (0.55, 0), fontsize=10, ha='center')
    ax3.annotate('PLOT HOLE', (0.65, 0), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    ax3.annotate('Recovery?', (0.85, 0), fontsize=10, ha='center')

    plt.tight_layout()
    plt.savefig('narrative_temporal_dynamics.png', dpi=150, bbox_inches='tight')
    print("✓ Temporal dynamics visualization saved to narrative_temporal_dynamics.png")
    plt.close()

def multi_agent_analysis():
    """
    Analyze how different audience types respond to the same narrative
    """
    print("\nAnalyzing multi-agent responses...")

    model = NarrativeImmersionField()

    # Define audience archetypes
    audiences = {
        'Casual Viewer': {'base_S': 0.9, 'C_sensitivity': 0.3, 'E_weight': 1.2},
        'Film Critic': {'base_S': 0.5, 'C_sensitivity': 1.5, 'E_weight': 0.8},
        'Genre Fan': {'base_S': 0.8, 'C_sensitivity': 0.5, 'E_weight': 1.0},
        'Child': {'base_S': 0.95, 'C_sensitivity': 0.1, 'E_weight': 1.3},
    }

    # Test scenario: Moderately coherent, highly emotional
    C, E, A = 0.6, 0.85, 0.75

    print(f"\nScenario: C={C}, E={E}, A={A}")
    print("-" * 60)

    for audience_name, params in audiences.items():
        # S depends on audience type and coherence
        S = params['base_S'] * (1 - params['C_sensitivity'] * (1 - C))
        S = np.clip(S, 0, 1)

        result = model.forward(C, E, S, A)
        state = model.classify_state(C, E, S, A)

        print(f"\n{audience_name}:")
        print(f"  Suspension of Disbelief: {S:.3f}")
        print(f"  Immersion Depth: {result['immersion_depth']:.3f}")
        print(f"  Critical Distance: {result['critical_distance']:.3f}")
        print(f"  State: {state}")

if __name__ == "__main__":
    print("="*60)
    print("ADVANCED NARRATIVE IMMERSION FIELD ANALYSIS")
    print("="*60)

    visualize_phase_space()
    temporal_dynamics_simulation()
    multi_agent_analysis()

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
