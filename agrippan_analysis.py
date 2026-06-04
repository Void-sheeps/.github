#!/usr/bin/env python3
"""
agrippan_analysis.py - Visualization for Agrippan Functor System
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from agrippan_functor import canonical_model

def run_analysis():
    print("Running Agrippan Functor Analysis...")

    model = canonical_model()
    relations = model["relations"]

    # Extract states
    states = [r["agrippan_state"] for r in relations]

    # Create DataFrame for plotting
    df = pd.DataFrame({"State": states})

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Count distribution
    ax = sns.countplot(data=df, x="State", palette="viridis", hue="State", legend=False)

    plt.title("Distribution of Agrippan States in Canonical Model")
    plt.ylabel("Count")
    plt.xlabel("Agrippan State")

    # Add counts on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.tight_layout()
    plt.savefig("agrippan_analysis.png")
    print(f"Analysis complete. Visualization saved to agrippan_analysis.png")

if __name__ == "__main__":
    run_analysis()
