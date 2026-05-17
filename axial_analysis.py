import torch
import matplotlib.pyplot as plt
from axial_transformer import AxialTransformerSystem, Domain, PerturbationOperator, DescriptionOperator

def run_analysis():
    print("--- Axial Transformer System Simulation ---")
    # 1. Setup
    torch.manual_seed(42)
    N = 100
    v = torch.linspace(-1, 1, N)
    initial_state = torch.exp(-v**2 / 0.1)  # Gaussian pulse
    domain = Domain(initial_state.unsqueeze(0)) # [B=1, N]

    # 2. Operators
    # Scale operator
    op_scale = PerturbationOperator(apply=lambda x: x * 1.5)
    # Inversion operator
    op_inv = PerturbationOperator(apply=lambda x: -x)
    # Shift operator
    op_shift = PerturbationOperator(apply=lambda x: x + 0.2)

    # 3. Description
    def describe(x):
        return f"Mean: {x.mean().item():.3f}, Std: {x.std().item():.3f}"
    desc_op = DescriptionOperator(describe=describe)

    # 4. Systems
    system_discrete = AxialTransformerSystem(desc_op, mode="discrete")
    system_field = AxialTransformerSystem(desc_op, mode="field")

    # Register operators with dummy embeddings
    # Using specific embeddings to see how attention affects weights
    emb_scale = torch.randn(64)
    emb_inv = torch.randn(64)
    emb_shift = torch.randn(64)

    system_discrete.register(op_scale, embedding=emb_scale)
    system_discrete.register(op_inv, embedding=emb_inv)
    system_discrete.register(op_shift, embedding=emb_shift)

    system_field.register(op_scale, embedding=emb_scale)
    system_field.register(op_inv, embedding=emb_inv)
    system_field.register(op_shift, embedding=emb_shift)

    # 5. Application
    # Base weights (prior)
    base_weights = torch.tensor([[0.5, 0.2, 0.3]])

    print("Applying Discrete Mode...")
    res_discrete = system_discrete.apply(domain, base_weights)
    print("Applying Field Mode...")
    res_field = system_field.apply(domain, base_weights)

    # 6. Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original vs Discrete
    axes[0, 0].plot(v.numpy(), initial_state.numpy(), 'k--', label="Original", alpha=0.5)
    axes[0, 0].plot(v.numpy(), res_discrete["state"][0].detach().numpy(), 'r', label="Discrete Result")
    axes[0, 0].set_title(f"Discrete Mode (Sequential)\n{res_discrete['description']}")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Original vs Field
    axes[0, 1].plot(v.numpy(), initial_state.numpy(), 'k--', label="Original", alpha=0.5)
    axes[0, 1].plot(v.numpy(), res_field["state"][0].detach().numpy(), 'b', label="Field Result")
    axes[0, 1].set_title(f"Field Mode (Superposition)\n{res_field['description']}")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Weights
    ops = ["Scale", "Invert", "Shift"]
    w_discrete = res_discrete["weights"][0].detach().numpy()
    w_field = res_field["weights"][0].detach().numpy()

    x_indices = torch.arange(len(ops)).numpy()
    width = 0.35

    axes[1, 0].bar(x_indices - width/2, w_discrete, width, color='red', alpha=0.6, label="Discrete Weights")
    axes[1, 0].bar(x_indices + width/2, w_field, width, color='blue', alpha=0.6, label="Field Weights")
    axes[1, 0].set_xticks(x_indices)
    axes[1, 0].set_xticklabels(ops)
    axes[1, 0].set_title("Recomputed Weights (Transformer Attention)")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(True, axis='y', alpha=0.3)

    # Axioms text
    from axial_transformer import AXIAL_FIELD_AXIOMS
    text = "Axial Field Axioms:\n\n"
    for k, v_val in AXIAL_FIELD_AXIOMS.items():
        if isinstance(v_val, dict):
            text += f"• {k.capitalize()}:\n"
            for sk, sv in v_val.items():
                text += f"    - {sk}: {sv}\n"
        else:
            text += f"• {k.capitalize()}: {v_val}\n"

    axes[1, 1].text(0.05, 0.95, text, transform=axes[1, 1].transAxes, fontsize=11,
                  verticalalignment='top', family='monospace',
                  bbox=dict(boxstyle='round,pad=1', facecolor='white', alpha=0.8, edgecolor='gray'))
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig("axial_analysis.png")
    print("Simulation complete. Saved axial_analysis.png")

if __name__ == "__main__":
    run_analysis()
