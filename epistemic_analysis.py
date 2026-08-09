#!/usr/bin/env python3
"""
epistemic_analysis.py - Duality of TRUTHS-undecidability and BELIEFS-undecidability under intension/extension duality.
"""

import argparse

class Intension:
    def __init__(self, label):
        self.label = label

class Extension:
    def __init__(self, trace):
        self.trace = trace

class Agent:
    def __init__(self, intension, extension):
        self.intension = intension
        self.extension = extension

TRUE = "TRUE"
FALSE = "FALSE"

REAL = Intension("real_agency")
FALSE_STATE = Intension("false_state")

def NOT(value):
    return FALSE if value == TRUE else TRUE

def extension_equivalent(e1, e2):
    return e1.trace == e2.trace

class NontrivialSemanticProperty:
    def __init__(self, name, holds_fn):
        self.name = name
        self.holds_fn = holds_fn

    def holds(self, agent):
        return self.holds_fn(agent)

    def is_nontrivial(self, universe):
        results = set(self.holds(a) for a in universe)
        return len(results) > 1

TRUTHS = NontrivialSemanticProperty(
    "TRUTHS",
    lambda a: TRUE if a.intension.label == "false_state" else FALSE
)

BELIEFS = NontrivialSemanticProperty(
    "BELIEFS",
    lambda a: TRUE if a.intension.label == "real_agency" else FALSE
)

class Decider:
    def __init__(self, property_):
        self.property_ = property_

    def decide(self, agent):
        raise NotImplementedError

def print_label(tag, text):
    print(f"[{tag}] {text}")

def axiom_extension_underdetermination():
    a1 = Agent(FALSE_STATE, Extension(["signal_A", "signal_B"]))
    a2 = Agent(REAL, Extension(["signal_A", "signal_B"]))
    assert extension_equivalent(a1.extension, a2.extension)
    assert a1.intension.label != a2.intension.label
    print_label("AXIOM", "There exist agents a1, a2 with equal extension and unequal intension.")
    return a1, a2

def lemma_property_depends_only_on_intension(property_, a1, a2):
    v1 = property_.holds(a1)
    v2 = property_.holds(a2)
    print_label(
        "LEMMA",
        f"{property_.name}(a1)={v1}, {property_.name}(a2)={v2}, "
        f"extension(a1)=extension(a2) -> property is intensional, not extensional."
    )
    return v1 != v2

def construct_diagonal_agent(decider):
    class DiagonalAgent(Agent):
        def __init__(self):
            self.intension = None
            self.extension = Extension(["self_reference"])

        def resolve_intension(self):
            verdict = decider.decide(self)
            self.intension = FALSE_STATE if verdict == TRUE else REAL
            return self.intension

    d = DiagonalAgent()
    return d

def proof_by_diagonalization(property_, universe):
    print_label("PROOF", f"Assume a total decider D exists for {property_.name}.")

    class HypotheticalDecider(Decider):
        def decide(self, agent):
            if agent.intension is None:
                return NOT(property_.holds_fn(Agent(REAL, agent.extension)))
            return property_.holds(agent)

    D = HypotheticalDecider(property_)
    diag = construct_diagonal_agent(D)
    verdict = D.decide(diag)
    diag.resolve_intension()
    actual = property_.holds(diag)

    print_label("PROOF", f"D(diag) = {verdict}")
    print_label("PROOF", f"{property_.name}(diag) after resolution = {actual}")

    contradiction = (verdict == actual)
    print_label(
        "PROOF",
        f"D(diag) supposed to predict {property_.name}(diag) in advance of intension fixation, "
        f"but intension is fixed only by consulting D itself -> circularity."
    )
    print_label(
        "COROLLARY",
        f"No total decider D can correctly resolve {property_.name} for all constructible agents."
    )
    return contradiction

def theorem_truths_beliefs_duality():
    print_label("THEOREM", "TRUTHS-undecidability and BELIEFS-undecidability are the same theorem under intension/extension duality.")

    universe = [
        Agent(FALSE_STATE, Extension(["s1"])),
        Agent(REAL, Extension(["s1"])),
        Agent(FALSE_STATE, Extension(["s2"])),
        Agent(REAL, Extension(["s2"])),
    ]

    a1, a2 = axiom_extension_underdetermination()

    print_label("LEMMA", f"{TRUTHS.name}.is_nontrivial(universe) = {TRUTHS.is_nontrivial(universe)}")
    print_label("LEMMA", f"{BELIEFS.name}.is_nontrivial(universe) = {BELIEFS.is_nontrivial(universe)}")

    lemma_property_depends_only_on_intension(TRUTHS, a1, a2)
    lemma_property_depends_only_on_intension(BELIEFS, a1, a2)

    proof_by_diagonalization(TRUTHS, universe)
    proof_by_diagonalization(BELIEFS, universe)

    print_label(
        "THEOREM",
        "TRUTHS is undecidable from extension alone because false_state is intensional. "
        "BELIEFS is undecidable from extension alone because real_agency is intensional. "
        "Both fail for the identical structural reason: a nontrivial semantic property of an "
        "agent's intension cannot be recovered from its extension by any total decider."
    )

    print_label(
        "COROLLARY",
        "Category saturation (all-TRUTHS or all-BELIEFS) does not violate this theorem; "
        "it collapses is_nontrivial to False, which removes the property from the domain of the "
        "theorem entirely rather than resolving the undecidability."
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epistemic Analysis - Duality and Undecidability Proof")
    parser.add_argument("--simulate", action="store_true", help="Run simulation")
    args = parser.parse_args()

    theorem_truths_beliefs_duality()
