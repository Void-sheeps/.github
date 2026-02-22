#!/usr/bin/env python3
"""
Musical Hash Field Analysis
===========================

Performs a detailed analysis of the mapping from 256-bit entropy to musical space.
Visualizes the distribution of musical parameters across a sample set.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from musical_hash_field import MusicalHashField

def run_analysis():
    print("Generating Musical Hash Field Analysis...")

    # Initialize model
    model = MusicalHashField()
    model.eval()

    # Generate 100 samples of 256-bit inputs
    num_samples = 100
    input_bits = torch.randint(0, 2, (num_samples, 256)).float()

    # Run model
    with torch.no_grad():
        outputs = model(input_bits)

    # Flatten outputs for distribution analysis
    pitches = outputs['pitch_class'].flatten().numpy()
    octaves = outputs['octave'].flatten().numpy()
    durations = outputs['duration'].flatten().numpy()

    # Plotting distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.histplot(pitches, bins=12, ax=axes[0], color='skyblue', discrete=True)
    axes[0].set_title("Pitch Class Distribution")
    axes[0].set_xticks(range(12))

    sns.histplot(octaves, bins=8, ax=axes[1], color='salmon', discrete=True)
    axes[1].set_title("Octave Distribution")
    axes[1].set_xticks(range(8))

    sns.histplot(durations, bins=4, ax=axes[2], color='green', discrete=True)
    axes[2].set_title("Duration Distribution")
    axes[2].set_xticks(range(4))

    plt.tight_layout()
    output_file = "musical_analysis.png"
    plt.savefig(output_file)
    print(f"Analysis saved to {output_file}")

if __name__ == "__main__":
    run_analysis()
