import torch
import torch.nn as nn
import torch.optim as optim

class NarrativeImmersionField(nn.Module):
    """
    A computational graph representing the Narrative Immersion Field.

    Inputs are continuous values [0, 1]:
    C: Narrative Coherence
    E: Emotional Resonance
    S: Suspension of Disbelief
    A: Attentional Capture
    """
    def __init__(self):
        super().__init__()
        # Thresholds defined in the JSON specification
        self.register_buffer('threshold_low', torch.tensor(0.2))
        self.register_buffer('threshold_mid', torch.tensor(0.4))
        self.register_buffer('threshold_high', torch.tensor(0.7))
        self.register_buffer('threshold_very_high', torch.tensor(0.8))

    def forward(self, C, E, S, A):
        """
        Calculates the derived metrics of the narrative field.
        """
        # Ensure inputs are clamped to [0, 1] for stability
        C = torch.clamp(C, 0, 1)
        E = torch.clamp(E, 0, 1)
        S = torch.clamp(S, 0, 1)
        A = torch.clamp(A, 0, 1)

        # --- Derived Functions ---
        # I_depth = C * E * S
        immersion_depth = C * E * S

        # N_int = C * E
        narrative_integrity = C * E

        # CD = (C * E) * (1 - S)
        critical_distance = narrative_integrity * (1 - S)

        return {
            "immersion_depth": immersion_depth,
            "narrative_integrity": narrative_integrity,
            "critical_distance": critical_distance,
            # Pass through original variables for state classification
            "vars": (C, E, S, A)
        }

    def classify_state(self, C, E, S, A):
        """
        Classifies the narrative state based on threshold logic.
        Returns a string label for the dominant state.
        """
        # We use standard python conditionals here for readability,
        # though this could be vectorized with boolean masks for large batches.

        val_C = C.item()
        val_E = E.item()
        val_S = S.item()
        val_A = A.item()

        # 1. Static Noise
        if val_C < 0.2 and val_E < 0.2:
            return "static_noise"

        # 2. Deep Flow Immersion (Optimal State)
        if val_C > 0.7 and val_E > 0.7 and val_S > 0.8:
            return "deep_flow_immersion"

        # 3. Hollow Spectacle
        if val_C > 0.8 and val_E < 0.2 and val_A > 0.8:
            return "hollow_spectacle"

        # 4. Dry Exposition
        if val_C > 0.7 and val_E < 0.3:
            return "dry_exposition"

        # 5. Dream Logic
        if val_C < 0.4 and val_E > 0.7 and val_S > 0.5:
            return "dream_logic"

        # 6. Uncanny Valley Risk (Stability Criterion)
        if val_C > 0.8 and val_E > 0.6 and val_S < 0.3:
            return "uncanny_valley_risk"

        return "liminal_state" # Undefined in spec

# ==========================================
# Simulation: "The Broken Spell"
# ==========================================

def simulate_narrative():
    print(f"{'='*60}\nNARRATIVE FIELD SIMULATION\n{'='*60}")

    model = NarrativeImmersionField()

    # Initialize a scenario: A High Budget Movie with a plot hole.
    # We require gradients to simulate "Phase Transitions" (fixing the story).
    # Initial State: Good visuals (A), Good Emotion (E), but Low Logic (C) -> "Dream Logic" or undefined

    # Story Parameters (Learnable)
    # Using logits + sigmoid to ensure [0,1] range during optimization
    story_params = nn.Parameter(torch.logit(torch.tensor([0.3, 0.8, 0.9]))) # C, E, A

    # Audience Parameter (Fixed for this agent)
    # This agent is willing to believe (S starts high), but S is reactive to C in our complex model
    # For this simple optimization, we assume S is a property of the audience interaction.
    audience_susceptibility = torch.tensor(0.85)

    optimizer = optim.Adam([story_params], lr=0.1)

    print("Phase 1: Optimizing for Deep Flow Immersion...")
    print("Goal: Increase Coherence (C) without losing Emotion (E)")

    for epoch in range(21):
        optimizer.zero_grad()

        # Extract current values
        current_vars = torch.sigmoid(story_params)
        C, E, A = current_vars[0], current_vars[1], current_vars[2]
        S = audience_susceptibility # In this model, S is fixed to the audience capability

        # Forward pass
        results = model(C, E, S, A)

        # Loss: We want to Maximize Immersion Depth
        # Loss = 1 - Immersion_Depth
        loss = 1.0 - results["immersion_depth"]

        # Backward pass (Phase Transition Force)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            state_name = model.classify_state(C, E, S, A)
            print(f"Step {epoch:02d} | C:{C:.2f} E:{E:.2f} S:{S:.2f} | Depth: {results['immersion_depth']:.3f} | State: {state_name}")

    # ==========================================
    # Phase 2: The Broken Spell (Plot Hole Event)
    # ==========================================
    print(f"\n{'-'*60}")
    print("Phase 2: EVENT - 'The Broken Spell'")
    print("A massive plot hole is revealed. Coherence (C) drops to 0.1.")
    print("According to specs, this causes a cascaded drop in Suspension of Disbelief (S).")
    print(f"{'-'*60}")

    with torch.no_grad():
        # Force drop C
        final_vars = torch.sigmoid(story_params)
        C_broken = torch.tensor(0.1)
        E_broken = final_vars[1] # Emotion remains?

        # Emergent logic: If C drops drastically, S crashes manually
        # (This simulates the agent realizing the movie is fake)
        S_broken = audience_susceptibility * 0.1
        A_broken = final_vars[2]

        broken_results = model(C_broken, E_broken, S_broken, A_broken)
        state_broken = model.classify_state(C_broken, E_broken, S_broken, A_broken)

        print(f"Post-Event | C:{C_broken:.2f} E:{E_broken:.2f} S:{S_broken:.2f} | Depth: {broken_results['immersion_depth']:.3f} | State: {state_broken}")

        # Check Critical Distance
        cd = broken_results["critical_distance"]
        print(f"Critical Distance (Analytical Viewing): {cd:.3f}")
        if cd > broken_results["immersion_depth"]:
            print(">> Result: The audience is now critiquing the script rather than experiencing it.")

if __name__ == "__main__":
    simulate_narrative()
