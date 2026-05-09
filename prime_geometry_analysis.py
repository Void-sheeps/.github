import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from symbolic_prime_geometry import SYMBOLS_EXT, encode, action, REV

def run_analysis():
    print("Running Prime Geometry Analysis...")

    # Data for Action Spectrum
    words = [
        ("bh",     list("bh"),     "L1 Binary"),
        ("abcg",   list("abcg"),   "L1 Nuclear"),
        ("def",    list("def"),    "Break 1"),
        ("efg",    list("efg"),    "Break 1b"),
        ("fgh",    list("fgh"),    "Break 2"),
        ("abcdek", list("abcdek"), "L2 Nuclear"),
    ]

    names = [w[0] for w in words]
    actions = [action(encode(w[1]))["A"].item() for w in words]
    roles = [w[2] for w in words]

    # Data for Binary Availability
    b_val = SYMBOLS_EXT["b"]
    all_primes = sorted(SYMBOLS_EXT.values())
    letters = [REV[p] for p in all_primes]
    has_binary = [((p - b_val) > 0 and (p - b_val) in SYMBOLS_EXT.values()) for p in all_primes]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # 1. Action Spectrum
    sns.barplot(x=names, y=actions, hue=roles, palette="viridis", ax=ax1, legend=True)
    ax1.set_title("Action Spectrum A(ω) across Prime Partitions")
    ax1.set_ylabel("Action Value A")
    ax1.set_ylim(0, max(actions) * 1.2)
    for i, v in enumerate(actions):
        ax1.text(i, v + 0.05, f"{v:.2f}", ha='center', fontweight='bold')

    # 2. Binary Availability (Goldbach b+q)
    binary_colors = ["#4c72b0" if h else "#c44e52" for h in has_binary]
    ax2.bar(letters, all_primes, color=binary_colors)
    ax2.set_title("Binary Availability (p = b + q) across Prime Space")
    ax2.set_ylabel("Prime Value")
    ax2.set_xlabel("Symbol")

    # Legend for binary availability
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4c72b0', label='Binary Available (p=b+q)'),
        Patch(facecolor='#c44e52', label='Binary Absent (Symmetry Broken)')
    ]
    ax2.legend(handles=legend_elements)

    # Annotate important points
    for i, (l, p) in enumerate(zip(letters, all_primes)):
        if l in ["i", "p"]:
            ax2.annotate(f"Target {l}={p}", xy=(i, p), xytext=(0, 10),
                         textcoords="offset points", ha='center', fontweight='bold',
                         arrowprops=dict(arrowstyle="->", color='black'))

    plt.tight_layout()
    output_path = "prime_geometry_analysis.png"
    plt.savefig(output_path)
    print(f"Analysis complete. Visualization saved to {output_path}")

if __name__ == "__main__":
    run_analysis()
