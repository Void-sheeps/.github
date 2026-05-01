"""
NeuralDOS
=========
A refined neuro-symbolic architecture that simulates a CPU-like execution model
with positional encodings, persistent recurrent state, and decoupled registers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- POSITIONAL ENCODING ---
class SinusoidalPE(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, dim]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# --- MEMORY ---
class Memory:
    def __init__(self, batch_size, seq_len, dim, device=None):
        self.state = torch.zeros(batch_size, seq_len, dim, device=device)

    def write(self, x):
        self.state = x

    def read(self):
        return self.state


# --- REGISTERS ---
class Registers:
    """
    AX: program-context accumulator.
        Opcode embeddings accumulate here across the program and iterations.
        INT 01h uses AX to modulate attention queries. Never overwritten,
        only additioned - encodes both instruction identity and iteration depth.

    BX: memory-summary register.
        Updated by INT 03h (global mean of current memory state). Separating
        AX and BX preserves the program-context signal while still making the
        global memory state available to INT 04h.

    h:  GRU hidden state, shape [num_layers, batch, dim].
        Persisted across fixed-point iterations so INT 02h builds recurrent
        context across the full convergence trajectory, not just within a
        single program pass.
    """
    def __init__(self, batch_size, dim, device=None):
        self.AX = torch.zeros(batch_size, 1, dim, device=device)
        self.BX = torch.zeros(batch_size, 1, dim, device=device)
        self.h = torch.zeros(1, batch_size, dim, device=device)


# --- KERNEL ---
class Kernel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)

    # INT 01h - Relation: AX-modulated self-attention.
    def int_01h_relation(self, mem, reg):
        x = mem.read()
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q + reg.AX  # AX preserved; only query is shifted
        attn_out = F.scaled_dot_product_attention(q, k, v)
        mem.write(x + attn_out)

    # INT 02h - Succession: GRU with persistent hidden state.
    def int_02h_succession(self, mem, reg):
        x = mem.read()
        out, h_new = self.rnn(x, reg.h)
        reg.h = h_new
        mem.write(out)

    # INT 03h - Load BX: global memory summary -> BX.
    def int_03h_load_bx(self, mem, reg):
        reg.BX = torch.mean(mem.read(), dim=1, keepdim=True)

    # INT 04h - Cross-feedback: BX -> memory via cross-attention.
    def int_04h_cross(self, mem, reg):
        x = mem.read()
        out, _ = self.cross_attn(query=x, key=reg.BX, value=reg.BX)
        mem.write(x + out)


# --- CPU ---
class CPU(nn.Module):
    def __init__(self, kernel, dim):
        super().__init__()
        self.kernel = kernel

        self.interrupt_table = {
            0x01: self.kernel.int_01h_relation,
            0x02: self.kernel.int_02h_succession,
            0x03: self.kernel.int_03h_load_bx,
            0x04: self.kernel.int_04h_cross,
        }

        self.program_embed = nn.Embedding(256, dim)

    def run(self, program_tensor, mem, reg):
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
        self.pe = SinusoidalPE(dim, max_len=seq_len)
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

        # Boot: token embeddings + positional encoding
        mem.write(self.pe(self.embed(x)))

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
        # Memory state + program context (AX) + memory summary (BX)
        out = mem.read() + reg.AX + reg.BX
        logits = self.head(out)

        # stack deltas: [iterations, batch]
        deltas = torch.stack(deltas)

        return logits, deltas
