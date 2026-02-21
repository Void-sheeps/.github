#!/usr/bin/env python3
"""
MusicalHashField
================

This module implements the MusicalHashField neural network, which maps
a 256-bit binary input into a sequence of 64 musical tokens.
Each token contains pitch, octave, direction, duration, and a silence flag.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

class MusicalHashField(nn.Module):
    """
    256-bit binary input -> 64 musical vector tokens

    Each token:
        pitch_class  : 0–11
        octave       : 0–7
        direction    : -1 or +1
        duration     : 0–3
        silence_flag : 0 or 1
    """

    def __init__(self):
        super().__init__()

        # projection layers
        self.pitch_proj = nn.Linear(4, 12)
        self.octave_proj = nn.Linear(4, 8)
        self.duration_proj = nn.Linear(4, 4)
        self.direction_proj = nn.Linear(4, 1)

    def forward(self, x):
        """
        x: [batch, 256] binary tensor
        """

        batch_size = x.shape[0]

        # reshape into 64 nibbles
        x = x.view(batch_size, 64, 4)

        # silence detection (all zeros in nibble)
        silence = (x.sum(dim=-1, keepdim=True) == 0).float()

        # pitch
        pitch_logits = self.pitch_proj(x)
        pitch = torch.argmax(pitch_logits, dim=-1)

        # octave
        octave_logits = self.octave_proj(x)
        octave = torch.argmax(octave_logits, dim=-1)

        # duration
        duration_logits = self.duration_proj(x)
        duration = torch.argmax(duration_logits, dim=-1)

        # direction
        direction_raw = torch.tanh(self.direction_proj(x))
        direction = torch.sign(direction_raw)

        return {
            "pitch_class": pitch,      # [B, 64]
            "octave": octave,          # [B, 64]
            "direction": direction,    # [B, 64]
            "duration": duration,      # [B, 64]
            "silence": silence.squeeze(-1)  # [B, 64]
        }

def run_simulation():
    print("--- MusicalHashField Simulation ---")

    # Initialize model
    model = MusicalHashField()
    model.eval()

    # Create a random 256-bit binary input (batch size 1)
    input_bits = torch.randint(0, 2, (1, 256)).float()

    # Run model
    with torch.no_grad():
        output = model(input_bits)

    print(f"Input bits shape: {input_bits.shape}")
    print("\nMusical Tokens (first 5):")
    for i in range(5):
        print(f"Token {i}:")
        print(f"  Pitch Class : {output['pitch_class'][0, i].item()}")
        print(f"  Octave      : {output['octave'][0, i].item()}")
        print(f"  Direction   : {output['direction'][0, i].item()}")
        print(f"  Duration    : {output['duration'][0, i].item()}")
        print(f"  Silence     : {output['silence'][0, i].item()}")

    print("\nSimulation complete: 256-bit entropy mapped to musical topology.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Musical Hash Field Simulation")
    parser.add_argument("--simulate", action="store_true", help="Run the model simulation")
    args = parser.parse_args()

    if args.simulate or not any(vars(args).values()):
        run_simulation()
