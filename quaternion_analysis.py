import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from functools import cmp_to_key

# Import functions from quaternion_confluence
from quaternion_confluence import (
    Rule, knuth_bendix, make_shortlex, discover_group_elements,
    normal_form, format_word
)

def run_analysis():
    print("Running Quaternion (Q_8) Confluence and Group Structure Analysis...")

    # 1. Setup alphabet and rules
    letter_order = ['a', 'b', 'A', 'B']
    cmp_fn = make_shortlex(letter_order)

    r0 = Rule(('a', 'a', 'a', 'a'), (), "a^4 = 1")
    r1 = Rule(('b', 'b'), ('a', 'a'), "b^2 = a^2")
    r2 = Rule(('a', 'b', 'a'), ('b',), "aba = b")

    inv_rules = [
        Rule(('a', 'A'), (), "a A = 1"),
        Rule(('A', 'a'), (), "A a = 1"),
        Rule(('b', 'B'), (), "b B = 1"),
        Rule(('B', 'b'), (), "B b = 1"),
    ]

    initial_rules = [r0, r1, r2] + inv_rules
    print("Completing rule system via Knuth-Bendix...")
    final_rules = knuth_bendix(initial_rules, cmp_fn, verbose=False)

    print("Discovering group elements...")
    elements = discover_group_elements(['a', 'b'], final_rules, cmp_fn)
    names = [format_word(el) for el in elements]
    elem_to_idx = {el: idx for idx, el in enumerate(elements)}

    # Generate multiplication table matrix
    n_elem = len(elements)
    mult_matrix = np.zeros((n_elem, n_elem), dtype=int)
    mult = {}
    inv_map = {}

    for i, u in enumerate(elements):
        for j, v in enumerate(elements):
            prod, _ = normal_form(u + v, final_rules)
            mult[(u, v)] = prod
            mult_matrix[i, j] = elem_to_idx[prod]
            if prod == ():
                inv_map[u] = v

    # Find subgroups
    subgroups = []
    for r in range(1, len(elements) + 1):
        for subset in itertools.combinations(elements, r):
            if () not in subset:
                continue
            closed = True
            for x in subset:
                for y in subset:
                    if mult[(x, y)] not in subset:
                        closed = False
                        break
                if not closed:
                    break
            if closed:
                subgroups.append(set(subset))

    # Verify normality for each subgroup
    subgroup_normality = []
    for H in subgroups:
        is_normal = True
        for g in elements:
            g_inv = inv_map[g]
            for h in H:
                gh = mult[(g, h)]
                ghg_inv = mult[(gh, g_inv)]
                if ghg_inv not in H:
                    is_normal = False
                    break
            if not is_normal:
                break
        subgroup_normality.append(is_normal)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Left Plot: Cayley Multiplication Table Heatmap
    sns.heatmap(
        mult_matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=names,
        yticklabels=names,
        ax=axes[0],
        cbar=True,
        square=True
    )
    # Overlay character representation labels as text annotations in cells
    for i in range(n_elem):
        for j in range(n_elem):
            target_elem = elements[mult_matrix[i, j]]
            target_name = format_word(target_elem)
            axes[0].text(
                j + 0.5, i + 0.3,
                f"\n({target_name})",
                color="black",
                ha="center",
                va="center",
                fontsize=9
            )

    axes[0].set_title("Cayley Multiplication Table Heatmap for Q_8", fontsize=14)
    axes[0].set_xlabel("Right Element (v)", fontsize=12)
    axes[0].set_ylabel("Left Element (u)", fontsize=12)

    # 2. Right Plot: Subgroup Size and Hamiltonian Proof Bar Chart
    subgroup_sizes = [len(H) for H in subgroups]
    subgroup_labels = []
    for idx, H in enumerate(subgroups, 1):
        h_sorted = sorted(list(H), key=cmp_to_key(cmp_fn))
        h_formatted = ", ".join(format_word(h) for h in h_sorted)
        subgroup_labels.append(f"H_{idx}\n(Size {len(H)})")

    colors = ['#2ecc71' if is_norm else '#e74c3c' for is_norm in subgroup_normality]
    bars = axes[1].bar(
        range(1, len(subgroups) + 1),
        subgroup_sizes,
        color=colors,
        edgecolor='black',
        alpha=0.85
    )
    axes[1].set_xticks(range(1, len(subgroups) + 1))
    axes[1].set_xticklabels(subgroup_labels, rotation=0, fontsize=10)
    axes[1].set_title("Normality & Sizes of all Subgroups in Q_8", fontsize=14)
    axes[1].set_xlabel("Subgroup Identifiers", fontsize=12)
    axes[1].set_ylabel("Subgroup Size (Order)", fontsize=12)
    axes[1].set_ylim(0, 9)

    # Add labels on top of bars
    for bar, is_norm in zip(bars, subgroup_normality):
        height = bar.get_height()
        status_text = "Normal" if is_norm else "Non-normal"
        axes[1].text(
            bar.get_x() + bar.get_width()/2.0,
            height + 0.2,
            status_text,
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color='darkgreen' if is_norm else 'darkred'
        )

    plt.suptitle("Confluence Analysis and Hamiltonian Proof of Quaternion Group Q_8", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = "quaternion_analysis.png"
    plt.savefig(output_path, dpi=150)
    print(f"Analysis complete. Visualization saved to '{output_path}'.")

if __name__ == "__main__":
    run_analysis()
