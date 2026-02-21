#!/usr/bin/env python3
"""
Reflective Watchdog Neural System
=================================

- Arquitetura reflexiva explícita
- Watchdog interno atualiza pesos com base em métricas de coerência
- Métricas: variância, entropia espectral, similaridade média
- Não-emergente: atualização arquitetural é explícita
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


# ---------------------------
# Base Model
# ---------------------------

class BaseModel(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transform = nn.Linear(d_model, d_model)
        nn.init.orthogonal_(self.transform.weight)  # preserva energia

    def forward(self, tokens):
        x = self.embedding(tokens)
        h = torch.tanh(self.transform(x))
        return h


# ---------------------------
# Watchdog Metrics
# ---------------------------

def spectral_entropy(h):
    """Entropia aproximada via autovalores singulares"""
    u, s, v = torch.svd(h.squeeze(0))
    p = s / (s.sum() + 1e-8)
    entropy = -(p * torch.log(p + 1e-8)).sum()
    return entropy

def mean_cosine_similarity(h):
    x_norm = F.normalize(h.squeeze(0), dim=-1)
    sim = torch.matmul(x_norm, x_norm.transpose(-2, -1))
    n = sim.size(-1)
    mask = ~torch.eye(n, dtype=bool)
    return sim[mask].mean()

def variance_metric(h):
    return h.var(dim=1).mean()


# ---------------------------
# Meta-Reflective Controller
# ---------------------------

class MetaReflectiveController(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.meta_gate = nn.Linear(d_model, 1)
        self.watchdog_lr = 0.1  # taxa de atualização interna dos pesos

    def forward(self, hidden, model):
        # Avaliação watchdog
        var = variance_metric(hidden)
        entropy = spectral_entropy(hidden)
        sim = mean_cosine_similarity(hidden)

        # Métrica combinada de coerência
        coherence = (var + entropy) / (sim + 1e-8)

        # Atualização reflexiva explícita (residual sobre pesos da camada linear)
        with torch.no_grad():
            grad_update = self.watchdog_lr * (1.0 / (coherence + 1e-8))
            model.transform.weight += grad_update * torch.randn_like(model.transform.weight)

        # Saída modulada
        meta_repr = hidden.mean(dim=1)
        gate = torch.sigmoid(self.meta_gate(meta_repr))
        logits = self.output_proj(hidden[:, -1]) * gate

        return logits, gate, {"variance": var, "entropy": entropy, "similarity": sim}


# ---------------------------
# Full Reflective System
# ---------------------------

class ReflectiveWatchdogModel(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.base = BaseModel(vocab_size, d_model)
        self.controller = MetaReflectiveController(d_model, vocab_size)

    def forward(self, tokens):
        hidden = self.base(tokens)
        logits, gate, metrics = self.controller(hidden, self.base)
        return {"logits": logits, "gate": gate, "metrics": metrics}


# ---------------------------
# Test / Simulation
# ---------------------------

if __name__ == "__main__":
    if "--simulate" in sys.argv:
        print("--- Reflective Watchdog Simulation ---")
        torch.manual_seed(42)
        vocab_size = 100
        d_model = 32
        seq_len = 12

        model = ReflectiveWatchdogModel(vocab_size, d_model)
        tokens = torch.randint(0, vocab_size, (1, seq_len))

        # Run several steps to see adaptation
        for i in range(5):
            output = model(tokens)
            print(f"Step {i} | Gate: {output['gate'].item():.4f} | Var: {output['metrics']['variance'].item():.4f} | Entropy: {output['metrics']['entropy'].item():.4f}")

        print("Simulation Complete.")
    else:
        torch.manual_seed(42)
        vocab_size = 100
        d_model = 32
        seq_len = 12

        model = ReflectiveWatchdogModel(vocab_size, d_model)
        tokens = torch.randint(0, vocab_size, (1, seq_len))

        output = model(tokens)
        print("Gate:", output["gate"].item())
        print("Variance:", output["metrics"]["variance"].item())
        print("Spectral Entropy:", output["metrics"]["entropy"].item())
        print("Mean Cosine Similarity:", output["metrics"]["similarity"].item())
