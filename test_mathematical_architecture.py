import pytest
from mathematical_architecture import (
    Axiom,
    Definition,
    Entity,
    EpistemicState,
    FormalSystem,
    InferenceRule,
    KnowledgeBase,
    KnowledgeItem,
    KnowledgeStatus,
    MathematicalArchitecture,
    MathematicalDomain,
    MathematicalEnvironment,
    MathematicalLayer,
    MathematicalModel,
    ModusPonens,
    Proof,
    ProofStep,
    Relation,
    Representation,
    RepresentationTransformation,
    RuleApplication,
    SemanticModel,
    Syllogism,
    Theorem,
    TheoremVerifier,
    build_environment,
    build_logical_model,
    build_mathematical_architecture,
)


def test_domain_and_layer_enums():
    assert MathematicalDomain.FOUNDATIONS.name == "FOUNDATIONS"
    assert MathematicalLayer.AXIOM.name == "AXIOM"
    assert KnowledgeStatus.VERIFIED.name == "VERIFIED"


def test_entities_and_relations():
    e = Entity(name="x", value=42, domain=MathematicalDomain.ALGEBRA)
    assert e.name == "x"
    assert e.value == 42
    assert e.domain == MathematicalDomain.ALGEBRA

    r = Relation(source="A", target="B", relation="isomorphism")
    assert r.source == "A"
    assert r.target == "B"
    assert r.relation == "isomorphism"


def test_representation():
    rep = Representation(language="formal")
    rel = Relation(source="X", target="Y", relation="maps_to")
    rep.add_symbol("f", "function")
    rep.add_relation(rel)

    assert rep.symbols["f"] == "function"
    assert rep.preserves(rel)
    assert not rep.preserves(Relation(source="A", target="B", relation="other"))


def test_inference_rules():
    mp = ModusPonens()
    assert mp.apply(["P -> Q", "P"]) == "Q"
    assert mp.apply(["P", "P -> Q"]) == "Q"
    assert mp.apply(["P", "Q"]) is None
    assert mp.apply(["P"]) is None

    syll = Syllogism()
    assert syll.apply(["A -> B", "B -> C", "A"]) == "C"
    assert syll.apply(["A", "A -> B", "B -> C"]) is None
    assert syll.apply(["A -> B", "B -> C"]) is None


class UnimplementedRule(InferenceRule):
    name = "Unimplemented"


def test_unimplemented_rule():
    rule = UnimplementedRule()
    with pytest.raises(NotImplementedError):
        rule.apply([])


def test_formal_system():
    fs = FormalSystem()
    fs.register_symbol("P")
    ax = Axiom(name="ax1", statement="P")
    df = Definition(
        name="def1", statement="D", definiendum="D", definiens="Definition body"
    )

    fs.register_axiom(ax)
    fs.register_definition(df)

    assert "P" in fs.symbols
    assert fs.has_axiom_statement("P")
    assert not fs.has_axiom_statement("Q")
    assert fs.has_definition_statement("D")
    assert not fs.has_definition_statement("E")


def test_knowledge_base():
    kb = KnowledgeBase()
    item = KnowledgeItem(
        name="item1",
        statement="P",
        layer=MathematicalLayer.AXIOM,
        status=KnowledgeStatus.ASSUMED,
    )
    kb.add(item)

    assert kb.contains_statement("P")
    assert not kb.contains_statement("Q")
    assert kb.get_by_statement("P") == item
    assert kb.get_by_statement("Q") is None

    item_refuted = KnowledgeItem(
        name="item2",
        statement="R",
        layer=MathematicalLayer.PROPOSITION,
        status=KnowledgeStatus.REFUTED,
    )
    kb.add(item_refuted)
    assert not kb.contains_statement("R")


def test_proof_and_verification():
    fs = FormalSystem()
    fs.register_rule(ModusPonens())
    kb = KnowledgeBase()

    kb.add(
        KnowledgeItem(
            name="ax1",
            statement="P",
            layer=MathematicalLayer.AXIOM,
            status=KnowledgeStatus.ASSUMED,
        )
    )
    kb.add(
        KnowledgeItem(
            name="ax2",
            statement="P -> Q",
            layer=MathematicalLayer.AXIOM,
            status=KnowledgeStatus.ASSUMED,
        )
    )

    verifier = TheoremVerifier(formal_system=fs, knowledge_base=kb)

    # Valid proof
    proof = Proof(theorem_name="Thm Q", statement="Q")
    proof.add_step(
        premises=["P", "P -> Q"], rule="Modus Ponens", conclusion="Q"
    )

    assert verifier.verify_proof(proof)
    assert proof.valid

    # Invalid proof - unknown rule
    proof_bad_rule = Proof(theorem_name="Thm Q", statement="Q")
    proof_bad_rule.add_step(
        premises=["P", "P -> Q"], rule="Unknown Rule", conclusion="Q"
    )
    assert not verifier.verify_proof(proof_bad_rule)

    # Invalid proof - premise not in knowledge base
    proof_missing_premise = Proof(theorem_name="Thm Z", statement="Z")
    proof_missing_premise.add_step(
        premises=["X", "X -> Z"], rule="Modus Ponens", conclusion="Z"
    )
    assert not verifier.verify_proof(proof_missing_premise)

    # Invalid proof - incorrect conclusion rule output mismatch
    proof_bad_conclusion = Proof(theorem_name="Thm Z", statement="Z")
    proof_bad_conclusion.add_step(
        premises=["P", "P -> Q"], rule="Modus Ponens", conclusion="Z"
    )
    assert not verifier.verify_proof(proof_bad_conclusion)

    # Invalid proof step verification single step method
    step = ProofStep(
        index=1,
        premises=["P", "P -> Q"],
        rule="Modus Ponens",
        conclusion="Q",
    )
    assert verifier.verify_step(step)

    bad_step = ProofStep(
        index=1,
        premises=["NOT_EXISTS"],
        rule="Modus Ponens",
        conclusion="Q",
    )
    assert not verifier.verify_step(bad_step)


def test_mathematical_model_prove():
    arch = build_mathematical_architecture()
    model = arch.create_model(
        domain=MathematicalDomain.FOUNDATIONS,
        language="logic",
        symbols={"P", "Q", "P -> Q"},
    )
    model.formal_system.register_rule(ModusPonens())

    model.register_axiom(Axiom(name="ax1", statement="P"))
    model.register_axiom(Axiom(name="ax2", statement="P -> Q"))

    # Prove valid theorem
    thm = Theorem(
        name="Thm1",
        statement="Q",
        proof=Proof(
            theorem_name="Thm1",
            statement="Q",
        ),
    )
    thm.proof.add_step(
        premises=["P", "P -> Q"], rule="Modus Ponens", conclusion="Q"
    )

    assert model.prove(thm)
    assert thm.status == KnowledgeStatus.VERIFIED

    # Unprovable theorem (no proof)
    thm_no_proof = Theorem(name="ThmNoProof", statement="Z", proof=None)
    assert not model.prove(thm_no_proof)
    assert thm_no_proof.status == KnowledgeStatus.REFUTED


def test_semantic_model():
    sem = SemanticModel(name="Sem1", domain=MathematicalDomain.ANALYSIS)
    sem.interpret("x", 10)
    sem.add_satisfied_statement("x > 0")

    assert sem.interpretation["x"] == 10
    assert sem.satisfies("x > 0")
    assert not sem.satisfies("x < 0")


def test_representation_transformation():
    r1 = Representation(language="L1")
    r2 = Representation(language="L2")
    rel = Relation(source="a", target="b", relation="r")
    trans = RepresentationTransformation(
        source=r1, target=r2, preserved_relations=[rel]
    )

    assert trans.preserves(rel)


def test_epistemic_state():
    state = EpistemicState()
    entity = Entity(name="E1")
    rel = Relation(source="E1", target="E1", relation="identity")

    state.add_entity(entity)
    state.add_relation(rel)

    rep = state.create_representation("lang")
    fs = state.create_formal_system({"sym"})
    sem = state.create_semantic_model("sem", MathematicalDomain.ALGEBRA)

    assert "E1" in state.entities
    assert rel in state.relations
    assert len(state.representations) == 1
    assert len(state.formal_systems) == 1
    assert len(state.semantic_models) == 1


def test_build_environment_export():
    env = build_environment()
    data = env.export()

    assert "domains" in data
    assert "dependency_structure" in data
    assert "models" in data
    assert len(data["models"]) == 1
    assert data["models"][0]["domain"] == "FOUNDATIONS"
    assert "Theorem Q" in data["models"][0]["theorems"]
    assert "Theorem R" in data["models"][0]["theorems"]


def test_rule_application_dataclass():
    ra = RuleApplication(
        rule_name="Modus Ponens",
        premises=("P", "P -> Q"),
        conclusion="Q",
    )
    assert ra.rule_name == "Modus Ponens"
    assert ra.premises == ("P", "P -> Q")
    assert ra.conclusion == "Q"
