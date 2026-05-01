"""
NeuralDOS
=========
A neuro-symbolic architecture that simulates a CPU-like execution model
using iterative fixed-point dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MEMORY ---
class Memory:
    def __init__(self, batch_size, seq_len, dim, device=None):
        # State will be overwritten by Boot sequence
        self.state = torch.zeros(batch_size, seq_len, dim, device=device)

    def write(self, x):
        self.state = x

    def read(self):
        return self.state


# --- REGISTERS ---
class Registers:
    def __init__(self, batch_size, dim, device=None):
        self.AX = torch.zeros(batch_size, 1, dim, device=device)
        self.BX = torch.zeros(batch_size, 1, dim, device=device)


# --- KERNEL ---
class Kernel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)

    # INT 01h: Relation (AX-modulated attention)
    def int_01h_relation(self, mem, reg):
        x = mem.read()
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q + reg.AX
        attn_out = F.scaled_dot_product_attention(q, k, v)

        mem.write(x + attn_out)

    # INT 02h: Succession
    def int_02h_succession(self, mem, reg):
        x = mem.read()
        out, _ = self.rnn(x)
        mem.write(out)

    # INT 03h: Load AX (global constraint)
    def int_03h_load_ax(self, mem, reg):
        reg.AX = torch.mean(mem.read(), dim=1, keepdim=True)

    # INT 04h: AX → Memory feedback
    def int_04h_cross(self, mem, reg):
        x = mem.read()
        ax = reg.AX
        out, _ = self.cross_attn(query=x, key=ax, value=ax)
        mem.write(x + out)


# --- CPU ---
class CPU(nn.Module):
    def __init__(self, kernel, dim):
        super().__init__()
        self.kernel = kernel

        self.interrupt_table = {
            0x01: self.kernel.int_01h_relation,
            0x02: self.kernel.int_02h_succession,
            0x03: self.kernel.int_03h_load_ax,
            0x04: self.kernel.int_04h_cross,
        }

        self.program_embed = nn.Embedding(256, dim)

    def run(self, program_tensor, mem, reg):
        # program_tensor should be pre-loaded as a tensor on the correct device
        for i in range(program_tensor.size(0)):
            opcode = program_tensor[i].item()
            opcode_idx = program_tensor[i]

            reg.AX = reg.AX + self.program_embed(opcode_idx).view(1, 1, -1)

            if opcode in self.interrupt_table:
                self.interrupt_table[opcode](mem, reg)


# --- MODEL ---
class NeuralDOS(nn.Module):
    def __init__(self, vocab_size, dim, seq_len, T=5, tol=1e-3):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.T = T                  # max iterations
        self.tol = tol              # convergence threshold

        self.embed = nn.Embedding(vocab_size, dim)
        self.kernel = Kernel(dim)
        self.processor = CPU(self.kernel, dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x, program):
        batch_size = x.size(0)
        device = x.device

        # Pre-convert program to tensor on device
        if not isinstance(program, torch.Tensor):
            program_tensor = torch.tensor(program, device=device, dtype=torch.long)
        else:
            program_tensor = program.to(device)

        mem = Memory(batch_size, self.seq_len, self.dim, device=device)
        reg = Registers(batch_size, self.dim, device=device)

        # Boot
        mem.write(self.embed(x))

        deltas = []
        prev_state = mem.read()

        # --- ITERATIVE EXECUTION (Fixed-Point Dynamics) ---
        for t in range(self.T):
            self.processor.run(program_tensor, mem, reg)

            current_state = mem.read()

            # stability measure
            delta = torch.norm(current_state - prev_state, dim=(1, 2))
            deltas.append(delta)

            # early stopping (all batch converged)
            if torch.all(delta < self.tol):
                break

            prev_state = current_state

        # --- READOUT ---
        out = mem.read() + reg.AX
        logits = self.head(out)

        # stack deltas: [iterations, batch]
        deltas = torch.stack(deltas)

        return logits, deltas
