from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Iterator
from collections import defaultdict
import hashlib, math, re


# ---------------------------------------------------------------------------
# Sub-symbolic compression
# ---------------------------------------------------------------------------

def token_entropy(text: str) -> float:
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0
    freq: defaultdict[str, int] = defaultdict(int)
    for t in tokens:
        freq[t] += 1
    n = len(tokens)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def subsymbolic_compress(text: str) -> np.ndarray:
    tokens = re.findall(r"\w+", text.lower())
    vocab = sorted(set(tokens))
    if not vocab:
        return np.zeros(8)
    index = {w: i for i, w in enumerate(vocab)}
    vec = np.zeros(len(vocab))
    for t in tokens:
        vec[index[t]] += 1
    vec /= vec.sum() + 1e-9
    out = np.zeros(8)
    for i, v in enumerate(vec):
        out[i % 8] += v
    norm = np.linalg.norm(out)
    return out / (norm + 1e-9)


def structural_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# Fix 1: phi scaled against realistic entropy range [1.5, 5.0]
_PHI_LOW, _PHI_HIGH = 1.5, 5.0

def entropy_to_phi(text: str) -> float:
    e = token_entropy(text)
    return float(np.clip((e - _PHI_LOW) / (_PHI_HIGH - _PHI_LOW), 0.0, 1.0))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class State:
    content: str
    phi: float = 0.0
    stability: float = 1.0
    vec: np.ndarray = field(default_factory=lambda: np.zeros(8))
    uid: str = field(default_factory=lambda: "")

    def __post_init__(self) -> None:
        self.vec = subsymbolic_compress(self.content)
        self.uid = structural_hash(self.content)
        if self.phi == 0.0:
            self.phi = entropy_to_phi(self.content)

    def in_domain(self) -> bool:
        return self.phi > 0.05 and self.stability > 0.05

    def __repr__(self) -> str:
        return (f"State(uid={self.uid}, φ={self.phi:.3f}, "
                f"stab={self.stability:.3f}, dom={self.in_domain()})")


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------

@dataclass
class Perturbation:
    name: str
    kind: str
    apply: Callable[[State], State]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def make_internalist_perturbations() -> list[Perturbation]:

    # Fix 2: repetition_density replaces unique_ratio
    # unique_ratio is near 1 for all non-repetitive prose — no signal.
    # repetition_density measures how much the text REUSES a small core vocab.
    def consistency_check(s: State) -> State:
        tokens = re.findall(r"\w+", s.content.lower())
        if not tokens:
            return s
        freq: defaultdict[str, int] = defaultdict(int)
        for t in tokens:
            freq[t] += 1
        # fraction of tokens that appear more than once
        repeated = sum(c for c in freq.values() if c > 1)
        repetition_density = repeated / (len(tokens) + 1e-9)
        # low repetition = high lexical diversity = stronger phi
        new_phi = s.phi * (0.6 + 0.4 * (1.0 - repetition_density))
        return State(s.content, phi=new_phi, stability=s.stability)

    def self_reference_pressure(s: State) -> State:
        self_tokens = {"i", "self", "system", "output", "process",
                       "def", "class", "return", "function", "method"}
        tokens = re.findall(r"\w+", s.content.lower())
        ratio = sum(1 for t in tokens if t in self_tokens) / (len(tokens) + 1e-9)
        new_stab = s.stability * math.exp(-2.0 * ratio)
        return State(s.content, phi=s.phi, stability=new_stab)

    def structural_invariance(s: State) -> State:
        words = s.content.split()
        if len(words) < 2:
            return s
        permuted = " ".join(words[::-1])
        s2 = State(permuted, phi=s.phi, stability=s.stability)
        dist = 1.0 - _cosine(s.vec, s2.vec)
        new_phi = s.phi * (1.0 - 0.4 * dist)
        return State(s.content, phi=new_phi, stability=s.stability)

    return [
        Perturbation("consistency", "internalist", consistency_check),
        Perturbation("self_reference", "internalist", self_reference_pressure),
        Perturbation("structural_invariance", "internalist", structural_invariance),
    ]


def make_externalist_perturbations(corpus: list[str]) -> list[Perturbation]:
    corpus_vecs = [subsymbolic_compress(c) for c in corpus if c.strip()]

    def vocabulary_extension(s: State) -> State:
        if not corpus_vecs:
            return s
        sims = [_cosine(s.vec, cv) for cv in corpus_vecs]
        max_sim = max(sims)
        new_phi = s.phi * (0.4 + 0.6 * max_sim)
        return State(s.content, phi=new_phi, stability=s.stability)

    # Fix 3: falsifiability uses cross-domain variance, not within-domain
    # variance must be computed across heterogeneous corpus partitions
    def falsifiability_pressure(s: State) -> State:
        if not corpus_vecs:
            return s
        sims = [_cosine(s.vec, cv) for cv in corpus_vecs]
        variance = float(np.var(sims))
        # rescale: variance 0.01 → reasonable signal; 0.05 → high
        rescaled = math.sqrt(np.clip(variance / 0.05, 0.0, 1.0))
        new_stab = s.stability * (0.3 + 0.7 * rescaled)
        return State(s.content, phi=s.phi, stability=float(np.clip(new_stab, 0, 1)))

    def domain_boundary(s: State) -> State:
        if not corpus_vecs:
            return s
        sims = [_cosine(s.vec, cv) for cv in corpus_vecs]
        mean_sim = float(np.mean(sims))
        boundary_proximity = abs(mean_sim - 0.5) * 2
        new_stab = s.stability * (0.6 + 0.4 * boundary_proximity)
        return State(s.content, phi=s.phi, stability=new_stab)

    return [
        Perturbation("vocabulary_extension", "externalist", vocabulary_extension),
        Perturbation("falsifiability", "externalist", falsifiability_pressure),
        Perturbation("domain_boundary", "externalist", domain_boundary),
    ]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class EpistemicAgent:
    def __init__(
        self,
        perturbations: list[Perturbation],
        history_limit: int = 128,
    ) -> None:
        self.perturbations = perturbations
        self.history: list[tuple[State, list[dict]]] = []
        self.history_limit = history_limit

    def _apply_all(self, s: State) -> tuple[State, list[dict]]:
        log: list[dict] = []
        current = s
        for p in self.perturbations:
            prev_phi, prev_stab = current.phi, current.stability
            current = p.apply(current)
            log.append({
                "perturbation": p.name,
                "kind": p.kind,
                "Δφ": round(current.phi - prev_phi, 4),
                "Δstab": round(current.stability - prev_stab, 4),
            })
        return current, log

    def step(self, s: State) -> State | None:
        s_out, log = self._apply_all(s)
        if len(self.history) < self.history_limit:
            self.history.append((s_out, log))
        return s_out if s_out.in_domain() else None

    def run(self, initial: State) -> Iterator[State]:
        current = initial
        while True:
            nxt = self.step(current)
            if nxt is None:
                break
            yield nxt
            current = nxt

    def confidence(self) -> float:
        if not self.history:
            return 0.0
        survived = [s for s, _ in self.history if s.in_domain()]
        resistance = len(survived) / len(self.history)
        phi_mean = float(np.mean([s.phi for s, _ in self.history]))
        stab_mean = float(np.mean([s.stability for s, _ in self.history]))
        return round((resistance * phi_mean * stab_mean) ** (1 / 3), 4)

    def report(self) -> dict:
        if not self.history:
            return {}
        phi_s = [s.phi for s, _ in self.history]
        stab_s = [s.stability for s, _ in self.history]
        int_d, ext_d = [], []
        for _, log in self.history:
            for e in log:
                delta = abs(e["Δφ"]) + abs(e["Δstab"])
                (int_d if e["kind"] == "internalist" else ext_d).append(delta)
        return {
            "steps": len(self.history),
            "confidence": self.confidence(),
            "phi_mean": round(float(np.mean(phi_s)), 4),
            "phi_var": round(float(np.var(phi_s)), 4),
            "stab_mean": round(float(np.mean(stab_s)), 4),
            "internalist_pressure": round(float(np.mean(int_d)) if int_d else 0, 4),
            "externalist_pressure": round(float(np.mean(ext_d)) if ext_d else 0, 4),
        }


# ---------------------------------------------------------------------------
# TerminalIterator
# ---------------------------------------------------------------------------

class TerminalIterator:
    def __init__(self, agent: EpistemicAgent, initial: State) -> None:
        self._gen = agent.run(initial)
        self._agent = agent
        self._last: State | None = None

    def __iter__(self) -> TerminalIterator:
        return self

    def __next__(self) -> State:
        s = next(self._gen)
        self._last = s
        return s

    def terminal_state(self) -> State | None:
        return self._last

    def epistemic_status(self) -> dict:
        return self._agent.report()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Heterogeneous corpus: Python domain + distant domains for variance signal
    corpus = [
        # Python lexicon
        "python keywords include def class if else for while import return lambda",
        "list comprehension offers concise way to create lists in python",
        "decorators are functions that modify other functions or methods",
        "python uses duck typing and dynamic typing with strong typing",
        "context managers with statement handles resource acquisition and release",
        "generators use yield to create iterators efficiently",
        "f strings provide formatted string literals since python 3.6",
        "type hints introduced in pep 484 improve code readability",
        "asyncio enables concurrent code using async and await keywords",
        "walrus operator assigns values inside expressions using colon equals",
        # Distant domains — needed for falsifiability variance
        "photosynthesis converts light energy into chemical energy stored in glucose",
        "the roman empire collapsed due to military overextension and economic strain",
        "sonata form consists of exposition development and recapitulation sections",
        "supply and demand curves intersect at the market equilibrium price",
        "plate tectonics describes the motion of lithospheric plates over the mantle",
        "the immune system distinguishes self from non-self through antigen recognition",
        "impressionist painters used loose brushwork to capture transient light effects",
        "neural networks learn by adjusting weights through gradient descent",
        "constitutional law governs the relationship between state power and individual rights",
        "thermodynamics describes energy transfer through heat work and entropy",
    ]

    internalist = make_internalist_perturbations()
    externalist = make_externalist_perturbations(corpus)
    agent = EpistemicAgent(internalist + externalist)

    python_seeds = [
        "def and class are fundamental keywords for defining functions and classes",
        "list comprehensions provide a pythonic way to transform iterables",
        "the with statement is used for context management and resource handling",
        "lambda creates anonymous functions for short single expression logic",
        "import and from are used to bring modules and names into namespace",
        "async def defines coroutine functions for asynchronous execution",
        "yield keyword turns a regular function into a generator iterator",
        "f strings are the modern preferred way to format strings in python",
        "type hints annotate variables and function signatures for static analysis",
        "walrus operator allows assignment within expressions using colon equals syntax",
    ]

    print(f"{'uid':>18}  {'φ':>6}  {'stab':>6}  {'dom':>5}  content[:60]")
    print("-" * 100)

    for seed in python_seeds:
        s0 = State(seed)
        it = TerminalIterator(agent, s0)
        for s in it:
            print(f"{s.uid:>18}  {s.phi:>6.3f}  {s.stability:>6.3f}"
                  f"  {str(s.in_domain()):>5}  {s.content[:60]}")

    print()
    rep = agent.report()
    for k, v in rep.items():
        print(f"  {k:>26}: {v}")
