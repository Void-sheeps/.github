import torch
import numpy as np
import matplotlib.pyplot as plt
from infinite_set import InfiniteSetEngine

def run_set_analysis():
    print("Initializing Infinite Set Analysis...")

    seed = 521171769
    depth = 5
    width = 10
    engine = InfiniteSetEngine(seed_int=seed, depth=depth, width=width)

    # Track evolution through levels
    levels_data = []
    current_x = torch.linspace(0.0, 1.0, width)
    levels_data.append(current_x.detach().numpy())

    with torch.no_grad():
        for i in range(depth):
            # Manually stepping through to capture internal states
            current_x = engine.relation_maps[i](current_x)
            current_x = torch.tanh(current_x)
            levels_data.append(current_x.detach().numpy())

        # Final absorption
        final_x = torch.sigmoid(current_x + engine.absorption_bias)
        levels_data.append(final_x.detach().numpy())

    # Visualization
    plt.figure(figsize=(12, 6))

    # 1. Heatmap of Level Evolution
    plt.subplot(1, 2, 1)
    evolution_matrix = np.array(levels_data)
    plt.imshow(evolution_matrix, aspect='auto', cmap='magma')
    plt.colorbar(label='Activation Intensity')
    plt.title("Hierarchical Vector Evolution")
    plt.xlabel("Set Element Index")
    plt.ylabel("Depth Level")
    plt.yticks(range(depth + 2), [f"Level {i}" for i in range(depth + 1)] + ["Absorption"])

    # 2. Convergence/Divergence Plot
    plt.subplot(1, 2, 2)
    for i in range(width):
        plt.plot(range(depth + 2), evolution_matrix[:, i], alpha=0.7, marker='o')

    plt.title("Element Propagation Path")
    plt.xlabel("Depth Level")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("infinite_set_analysis.png")
    print("Analysis complete. Saved to infinite_set_analysis.png")

if __name__ == "__main__":
    run_set_analysis()
