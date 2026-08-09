import pytest
from epistemic_analysis import (
    Intension,
    Extension,
    Agent,
    NOT,
    TRUE,
    FALSE,
    REAL,
    FALSE_STATE,
    extension_equivalent,
    NontrivialSemanticProperty,
    TRUTHS,
    BELIEFS,
    axiom_extension_underdetermination,
    lemma_property_depends_only_on_intension,
    proof_by_diagonalization,
    theorem_truths_beliefs_duality,
)

def test_basic_structures():
    i = Intension("test_intension")
    assert i.label == "test_intension"

    ext = Extension(["trace_1", "trace_2"])
    assert ext.trace == ["trace_1", "trace_2"]

    agent = Agent(i, ext)
    assert agent.intension == i
    assert agent.extension == ext

def test_not_operator():
    assert NOT(TRUE) == FALSE
    assert NOT(FALSE) == TRUE

def test_extension_equivalence():
    e1 = Extension(["a", "b"])
    e2 = Extension(["a", "b"])
    e3 = Extension(["a", "c"])
    assert extension_equivalent(e1, e2) is True
    assert extension_equivalent(e1, e3) is False

def test_nontrivial_semantic_property():
    prop = NontrivialSemanticProperty("PROP", lambda a: TRUE if a.intension.label == "yes" else FALSE)

    agent_yes = Agent(Intension("yes"), Extension([]))
    agent_no = Agent(Intension("no"), Extension([]))

    assert prop.holds(agent_yes) == TRUE
    assert prop.holds(agent_no) == FALSE

    universe = [agent_yes, agent_no]
    assert prop.is_nontrivial(universe) is True

    universe_trivial = [agent_yes, agent_yes]
    assert prop.is_nontrivial(universe_trivial) is False

def test_axiom_underdetermination():
    a1, a2 = axiom_extension_underdetermination()
    assert extension_equivalent(a1.extension, a2.extension) is True
    assert a1.intension.label != a2.intension.label

def test_lemma_property_depends_only_on_intension():
    a1, a2 = axiom_extension_underdetermination()
    assert lemma_property_depends_only_on_intension(TRUTHS, a1, a2) is True
    assert lemma_property_depends_only_on_intension(BELIEFS, a1, a2) is True

def test_proof_by_diagonalization():
    universe = [
        Agent(FALSE_STATE, Extension(["s1"])),
        Agent(REAL, Extension(["s1"])),
    ]
    # TRUTHS returns True because verdict and actual are equal
    assert proof_by_diagonalization(TRUTHS, universe) is True
    # BELIEFS returns False because verdict (FALSE) and actual (TRUE) differ
    assert proof_by_diagonalization(BELIEFS, universe) is False

def test_theorem_duality():
    theorem_truths_beliefs_duality()
