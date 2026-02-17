import torch
import numpy as np
import matplotlib.pyplot as plt
from hex_engine import CloverPitHexEngine

def run_analysis():
    print("Initializing Hex Engine Analysis...")
    engine = CloverPitHexEngine()

    # 1. Visualize the Hex Grid / Charme Matrix
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    charme_data = engine.charme_matrix.detach().numpy()
    im = plt.imshow(charme_data, cmap="YlOrRd")
    plt.colorbar(im, label='Bias Strength')

    # Add text annotations
    for i in range(charme_data.shape[0]):
        for j in range(charme_data.shape[1]):
            plt.text(j, i, f'{charme_data[i, j]:.1f}', ha="center", va="center", color="black")

    plt.title("Charme Matrix (EmberRune 4D)")
    plt.xlabel("Hex Columns")
    plt.ylabel("Hex Rows")

    # 2. Spin Sensitivity Analysis (0ms to 10ms)
    ms_range = torch.linspace(0, 10, 100)
    seed_data = torch.tensor([0.05, 0.21, 0.17, 0.17, 0.69])

    results = []
    with torch.no_grad():
        for ms in ms_range:
            output = engine(seed_data, ms.unsqueeze(0))
            results.append(output.item())

    plt.subplot(1, 2, 2)
    plt.plot(ms_range.numpy(), results, color='red', lw=2)
    plt.fill_between(ms_range.numpy(), results, alpha=0.3, color='red')
    plt.axhline(y=0.95, color='gray', linestyle='--', label='Success Threshold')
    plt.title("Collapse Probability vs. Spin (MS)")
    plt.xlabel("Input MS (Temporal Offset)")
    plt.ylabel("Win Probability")
    plt.ylim(0, 1.1)
    plt.legend()

    plt.tight_layout()
    plt.savefig("hex_engine_analysis.png")
    print("Analysis complete. Saved to hex_engine_analysis.png")

if __name__ == "__main__":
    run_analysis()
