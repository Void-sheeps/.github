import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from symbolic_prime_geometry import SYMBOLS, decompose, REV

def run_analysis():
    print("Running Extended Prime Geometry Analysis...")

    words = [
        ("bh",     list("bh")),
        ("abcg",   list("abcg")),
        ("def",    list("def")),
        ("efg",    list("efg")),
        ("fgh",    list("fgh")),
        ("abcdek", list("abcdek")),
    ]

    names = []
    alphas = []
    betas = []
    phis = []
    moduli = []
    binaries = []
    targets = []

    for name, w in words:
        r = decompose(w)
        names.append(name)
        alphas.append(r["alpha"])
        betas.append(r["beta"])
        phis.append(r["phi"])
        moduli.append(r["modulus"])
        binaries.append(r["binary"])
        targets.append(r["target"])

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    # 1. Action Space (alpha, beta) Plane
    ax1 = axes[0]
    colors = ["#4c72b0" if b else "#c44e52" for b in binaries]
    ax1.scatter(alphas, betas, c=colors, s=100, edgecolors='black', zorder=5)

    # Draw arc for modulus
    max_m = max(moduli)
    theta = np.linspace(0, np.pi/2, 100)
    for m in sorted(list(set(moduli))):
        ax1.plot(m * np.cos(theta), m * np.sin(theta), 'k--', alpha=0.2)

    for i, name in enumerate(names):
        ax1.annotate(f" {name}", (alphas[i], betas[i]), fontweight='bold')

    ax1.set_title("V = S₁ ⊕ S_b Action Space")
    ax1.set_xlabel("α (S₁ Nucleus Content)")
    ax1.set_ylabel("β (S_b Break Content)")
    ax1.set_xlim(-0.1, max(alphas) + 0.5)
    ax1.set_ylim(-0.1, max(betas) + 0.5)
    ax1.grid(True, alpha=0.3)

    # 2. Phase Angle vs Binary Availability
    ax2 = axes[1]
    y_bin = [1 if b else 0 for b in binaries]
    sns.scatterplot(x=phis, y=y_bin, hue=names, s=200, ax=ax2, palette="deep")
    ax2.set_title("Phase Angle φ vs Binary Availability")
    ax2.set_xlabel("Phase Angle φ (degrees)")
    ax2.set_ylabel("Binary Available (1=Yes, 0=No)")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["✗ Non-invertible", "✓ Invertible"])
    ax2.set_xlim(-5, 95)
    ax2.grid(True, alpha=0.3)

    # 3. Modulus Spectrum
    ax3 = axes[2]
    sns.barplot(x=names, y=moduli, hue=names, palette="viridis", ax=ax3, legend=False)
    ax3.set_title("Word Modulus |ω| in V")
    ax3.set_ylabel("Modulus Value")
    for i, v in enumerate(moduli):
        ax3.text(i, v + 0.05, f"{v:.2f}", ha='center')

    # 4. Target Prime vs Phase Angle
    ax4 = axes[3]
    sns.scatterplot(x=phis, y=targets, size=moduli, hue=binaries, sizes=(50, 400), ax=ax4)
    for i, name in enumerate(names):
        ax4.annotate(f" {REV[targets[i]]}", (phis[i], targets[i]))

    ax4.set_title("Target Prime vs Phase Angle φ")
    ax4.set_xlabel("Phase Angle φ")
    ax4.set_ylabel("Target Prime Value")
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Symbolic Prime Geometry: Field-Like Extension Analysis", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = "prime_geometry_analysis.png"
    plt.savefig(output_path)
    print(f"Analysis complete. Visualization saved to {output_path}")

if __name__ == "__main__":
    run_analysis()
