from __future__ import annotations

import pprint
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


class MathematicalDomain(Enum):
    FOUNDATIONS = auto()
    ALGEBRA = auto()
    ANALYSIS = auto()
    GEOMETRY_AND_TOPOLOGY = auto()
    APPLIED_MATHEMATICS = auto()


class MathematicalLayer(Enum):
    AXIOM = auto()
    DEFINITION = auto()
    PROPOSITION = auto()
    THEOREM = auto()
    MODEL = auto()


class KnowledgeStatus(Enum):
    ASSUMED = auto()
    DEFINED = auto()
    DERIVED = auto()
    VERIFIED = auto()
    REFUTED = auto()


@dataclass(frozen=True)
class Entity:
    name: str
    value: Any = None
    domain: Optional[MathematicalDomain] = None


@dataclass(frozen=True)
class Relation:
    source: str
    target: str
    relation: str
    invariant: Optional[str] = None
    transformation: Optional[str] = None


@dataclass
class Representation:
    language: str
    symbols: Dict[str, str] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)

    def add_symbol(self, symbol: str, meaning: str) -> None:
        self.symbols[symbol] = meaning

    def add_relation(self, relation: Relation) -> None:
        self.relations.append(relation)

    def preserves(self, relation: Relation) -> bool:
        return relation in self.relations


@dataclass
class Definition:
    name: str
    statement: str
    definiendum: str
    definiens: str
    status: KnowledgeStatus = KnowledgeStatus.DEFINED


@dataclass
class Axiom:
    name: str
    statement: str
    status: KnowledgeStatus = KnowledgeStatus.ASSUMED


@dataclass
class Proposition:
    name: str
    statement: str
    premises: List[str] = field(default_factory=list)
    status: KnowledgeStatus = KnowledgeStatus.DERIVED


@dataclass(frozen=True)
class RuleApplication:
    rule_name: str
    premises: Tuple[str, ...]
    conclusion: str


class InferenceRule:
    name: str

    def apply(self, premises: List[str]) -> Optional[str]:
        raise NotImplementedError


class ModusPonens(InferenceRule):
    name = "Modus Ponens"

    def apply(self, premises: List[str]) -> Optional[str]:
        if len(premises) != 2:
            return None

        first, second = premises

        if " -> " in first:
            antecedent, consequent = first.split(" -> ", 1)
            if antecedent == second:
                return consequent

        if " -> " in second:
            antecedent, consequent = second.split(" -> ", 1)
            if antecedent == first:
                return consequent

        return None


class Syllogism(InferenceRule):
    name = "Syllogism"

    def apply(self, premises: List[str]) -> Optional[str]:
        if len(premises) != 3:
            return None

        first, second, third = premises

        if " -> " not in first or " -> " not in second:
            return None

        a, b = first.split(" -> ", 1)
        b2, c = second.split(" -> ", 1)

        if b == b2 and third == a:
            return c

        if a == third and b == b2:
            return c

        return None


@dataclass
class FormalSystem:
    symbols: Set[str] = field(default_factory=set)
    axioms: Dict[str, Axiom] = field(default_factory=dict)
    definitions: Dict[str, Definition] = field(default_factory=dict)
    rules: Dict[str, InferenceRule] = field(default_factory=dict)

    def register_symbol(self, symbol: str) -> None:
        self.symbols.add(symbol)

    def register_axiom(self, axiom: Axiom) -> None:
        self.axioms[axiom.name] = axiom

    def register_definition(self, definition: Definition) -> None:
        self.definitions[definition.name] = definition

    def register_rule(self, rule: InferenceRule) -> None:
        self.rules[rule.name] = rule

    def has_axiom_statement(self, statement: str) -> bool:
        return any(
            axiom.statement == statement
            for axiom in self.axioms.values()
        )

    def has_definition_statement(self, statement: str) -> bool:
        return any(
            definition.statement == statement
            for definition in self.definitions.values()
        )


@dataclass
class KnowledgeItem:
    name: str
    statement: str
    layer: MathematicalLayer
    status: KnowledgeStatus
    source: Optional[str] = None


@dataclass
class KnowledgeBase:
    items: Dict[str, KnowledgeItem] = field(default_factory=dict)

    def add(self, item: KnowledgeItem) -> None:
        self.items[item.name] = item

    def contains_statement(self, statement: str) -> bool:
        return any(
            item.statement == statement
            and item.status in {
                KnowledgeStatus.ASSUMED,
                KnowledgeStatus.DEFINED,
                KnowledgeStatus.DERIVED,
                KnowledgeStatus.VERIFIED,
            }
            for item in self.items.values()
        )

    def get_by_statement(self, statement: str) -> Optional[KnowledgeItem]:
        for item in self.items.values():
            if item.statement == statement:
                return item
        return None


@dataclass
class ProofStep:
    index: int
    premises: List[str]
    rule: str
    conclusion: str
    valid: bool = False


@dataclass
class Proof:
    theorem_name: str
    statement: str
    steps: List[ProofStep] = field(default_factory=list)
    valid: bool = False

    def add_step(
        self,
        premises: List[str],
        rule: str,
        conclusion: str,
        valid: bool = False,
    ) -> ProofStep:
        step = ProofStep(
            index=len(self.steps) + 1,
            premises=premises,
            rule=rule,
            conclusion=conclusion,
            valid=valid,
        )
        self.steps.append(step)
        return step


@dataclass
class Theorem:
    name: str
    statement: str
    proof: Optional[Proof] = None
    status: KnowledgeStatus = KnowledgeStatus.DERIVED


@dataclass
class TheoremVerifier:
    formal_system: FormalSystem
    knowledge_base: KnowledgeBase

    def verify_premises(self, premises: Iterable[str]) -> bool:
        return all(
            self.knowledge_base.contains_statement(premise)
            for premise in premises
        )

    def verify_rule_application(
        self,
        premises: List[str],
        rule_name: str,
        expected_conclusion: str,
    ) -> bool:
        rule = self.formal_system.rules.get(rule_name)

        if rule is None:
            return False

        derived = rule.apply(premises)

        return derived == expected_conclusion

    def verify_step(self, step: ProofStep) -> bool:
        premises_valid = self.verify_premises(step.premises)

        if not premises_valid:
            step.valid = False
            return False

        step.valid = self.verify_rule_application(
            step.premises,
            step.rule,
            step.conclusion,
        )

        return step.valid

    def verify_proof(self, proof: Proof) -> bool:
        derived_statements = {
            item.statement
            for item in self.knowledge_base.items.values()
            if item.status in {
                KnowledgeStatus.ASSUMED,
                KnowledgeStatus.DEFINED,
                KnowledgeStatus.DERIVED,
                KnowledgeStatus.VERIFIED,
            }
        }

        for step in proof.steps:
            premises_available = all(
                premise in derived_statements
                for premise in step.premises
            )

            if not premises_available:
                step.valid = False
                proof.valid = False
                return False

            if not self.verify_rule_application(
                step.premises,
                step.rule,
                step.conclusion,
            ):
                step.valid = False
                proof.valid = False
                return False

            step.valid = True
            derived_statements.add(step.conclusion)

        proof.valid = bool(proof.steps) and (
            proof.steps[-1].conclusion == proof.statement
        )

        return proof.valid


@dataclass
class MathematicalModel:
    domain: MathematicalDomain
    representation: Representation
    formal_system: FormalSystem
    knowledge_base: KnowledgeBase = field(default_factory=KnowledgeBase)
    theorems: Dict[str, Theorem] = field(default_factory=dict)

    def register_axiom(self, axiom: Axiom) -> None:
        self.formal_system.register_axiom(axiom)
        self.knowledge_base.add(
            KnowledgeItem(
                name=axiom.name,
                statement=axiom.statement,
                layer=MathematicalLayer.AXIOM,
                status=KnowledgeStatus.ASSUMED,
            )
        )

    def register_definition(self, definition: Definition) -> None:
        self.formal_system.register_definition(definition)
        self.knowledge_base.add(
            KnowledgeItem(
                name=definition.name,
                statement=definition.statement,
                layer=MathematicalLayer.DEFINITION,
                status=KnowledgeStatus.DEFINED,
            )
        )

    def prove(self, theorem: Theorem) -> bool:
        if theorem.proof is None:
            theorem.status = KnowledgeStatus.REFUTED
            return False

        verifier = TheoremVerifier(
            formal_system=self.formal_system,
            knowledge_base=self.knowledge_base,
        )

        valid = verifier.verify_proof(theorem.proof)

        if valid:
            theorem.status = KnowledgeStatus.VERIFIED

            self.knowledge_base.add(
                KnowledgeItem(
                    name=theorem.name,
                    statement=theorem.statement,
                    layer=MathematicalLayer.THEOREM,
                    status=KnowledgeStatus.VERIFIED,
                    source=theorem.proof.theorem_name,
                )
            )
        else:
            theorem.status = KnowledgeStatus.REFUTED

        self.theorems[theorem.name] = theorem

        return valid


@dataclass
class SemanticModel:
    name: str
    domain: MathematicalDomain
    interpretation: Dict[str, Any] = field(default_factory=dict)
    satisfied_statements: Set[str] = field(default_factory=set)

    def interpret(self, symbol: str, value: Any) -> None:
        self.interpretation[symbol] = value

    def satisfies(self, statement: str) -> bool:
        return statement in self.satisfied_statements

    def add_satisfied_statement(self, statement: str) -> None:
        self.satisfied_statements.add(statement)


@dataclass
class RepresentationTransformation:
    source: Representation
    target: Representation
    mapping: Dict[str, str] = field(default_factory=dict)
    preserved_relations: List[Relation] = field(default_factory=list)

    def preserves(self, relation: Relation) -> bool:
        return relation in self.preserved_relations


@dataclass
class EpistemicState:
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    representations: List[Representation] = field(default_factory=list)
    formal_systems: List[FormalSystem] = field(default_factory=list)
    models: List[MathematicalModel] = field(default_factory=list)
    semantic_models: List[SemanticModel] = field(default_factory=list)

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.name] = entity

    def add_relation(self, relation: Relation) -> None:
        self.relations.append(relation)

    def create_representation(
        self,
        language: str,
    ) -> Representation:
        representation = Representation(
            language=language,
            relations=list(self.relations),
        )
        self.representations.append(representation)
        return representation

    def create_formal_system(
        self,
        symbols: Optional[Set[str]] = None,
    ) -> FormalSystem:
        system = FormalSystem(
            symbols=symbols or set(),
        )
        self.formal_systems.append(system)
        return system

    def create_semantic_model(
        self,
        name: str,
        domain: MathematicalDomain,
    ) -> SemanticModel:
        model = SemanticModel(
            name=name,
            domain=domain,
        )
        self.semantic_models.append(model)
        return model


@dataclass
class MathematicalArchitecture:
    domains: Dict[MathematicalDomain, List[str]] = field(default_factory=dict)
    epistemic_state: EpistemicState = field(
        default_factory=EpistemicState
    )

    def register_domain(
        self,
        domain: MathematicalDomain,
        subtopics: List[str],
    ) -> None:
        self.domains[domain] = subtopics

    def create_model(
        self,
        domain: MathematicalDomain,
        language: str,
        symbols: Optional[Set[str]] = None,
    ) -> MathematicalModel:
        representation = self.epistemic_state.create_representation(
            language
        )

        formal_system = self.epistemic_state.create_formal_system(
            symbols
        )

        model = MathematicalModel(
            domain=domain,
            representation=representation,
            formal_system=formal_system,
        )

        self.epistemic_state.models.append(model)

        return model

    def dependency_structure(self) -> Dict[str, Tuple[str, ...]]:
        return {
            "syntactic": (
                "symbol",
                "axiom",
                "definition",
                "rule",
                "derivation",
                "proposition",
                "theorem",
            ),
            "semantic": (
                "representation",
                "interpretation",
                "model",
                "satisfaction",
            ),
            "epistemic": (
                "entity",
                "relation",
                "representation",
                "formalization",
                "derivation",
                "verification",
                "knowledge",
            ),
        }


@dataclass
class MathematicalEnvironment:
    architecture: MathematicalArchitecture
    active_models: List[MathematicalModel] = field(default_factory=list)
    external_context: Dict[str, Any] = field(default_factory=dict)

    def add_model(self, model: MathematicalModel) -> None:
        self.active_models.append(model)

    def export(self) -> Dict[str, Any]:
        return {
            "domains": {
                domain.name: subtopics
                for domain, subtopics in self.architecture.domains.items()
            },
            "dependency_structure": (
                self.architecture.dependency_structure()
            ),
            "models": [
                {
                    "domain": model.domain.name,
                    "language": model.representation.language,
                    "symbols": sorted(model.formal_system.symbols),
                    "axioms": {
                        name: axiom.statement
                        for name, axiom
                        in model.formal_system.axioms.items()
                    },
                    "definitions": {
                        name: definition.statement
                        for name, definition
                        in model.formal_system.definitions.items()
                    },
                    "theorems": {
                        name: theorem.statement
                        for name, theorem
                        in model.theorems.items()
                    },
                }
                for model in self.active_models
            ],
            "external_context": self.external_context,
        }


def build_mathematical_architecture() -> MathematicalArchitecture:
    architecture = MathematicalArchitecture()

    architecture.register_domain(
        MathematicalDomain.FOUNDATIONS,
        [
            "set theory",
            "mathematical logic",
            "category theory",
            "proof theory",
            "model theory",
        ],
    )

    architecture.register_domain(
        MathematicalDomain.ALGEBRA,
        [
            "linear algebra",
            "abstract algebra",
            "number theory",
            "algebraic geometry",
        ],
    )

    architecture.register_domain(
        MathematicalDomain.ANALYSIS,
        [
            "real analysis",
            "complex analysis",
            "functional analysis",
            "differential equations",
        ],
    )

    architecture.register_domain(
        MathematicalDomain.GEOMETRY_AND_TOPOLOGY,
        [
            "euclidean geometry",
            "differential geometry",
            "algebraic topology",
            "general topology",
        ],
    )

    architecture.register_domain(
        MathematicalDomain.APPLIED_MATHEMATICS,
        [
            "numerical analysis",
            "optimization",
            "mathematical physics",
            "probability",
            "statistics",
        ],
    )

    return architecture


def build_logical_model(
    architecture: MathematicalArchitecture,
) -> MathematicalModel:
    model = architecture.create_model(
        domain=MathematicalDomain.FOUNDATIONS,
        language="propositional logic",
        symbols={"P", "Q", "R", "->"},
    )

    model.formal_system.register_rule(ModusPonens())
    model.formal_system.register_rule(Syllogism())

    model.register_axiom(
        Axiom(
            name="Axiom P",
            statement="P",
        )
    )

    model.register_axiom(
        Axiom(
            name="Implication PQ",
            statement="P -> Q",
        )
    )

    theorem_q = Theorem(
        name="Theorem Q",
        statement="Q",
        proof=Proof(
            theorem_name="Theorem Q",
            statement="Q",
        ),
    )

    theorem_q.proof.add_step(
        premises=["P", "P -> Q"],
        rule="Modus Ponens",
        conclusion="Q",
    )

    model.prove(theorem_q)

    model.register_axiom(
        Axiom(
            name="Implication QR",
            statement="Q -> R",
        )
    )

    theorem_r = Theorem(
        name="Theorem R",
        statement="R",
        proof=Proof(
            theorem_name="Theorem R",
            statement="R",
        ),
    )

    theorem_r.proof.add_step(
        premises=["Q", "Q -> R"],
        rule="Modus Ponens",
        conclusion="R",
    )

    model.prove(theorem_r)

    return model


def build_environment() -> MathematicalEnvironment:
    architecture = build_mathematical_architecture()

    logical_model = build_logical_model(
        architecture
    )

    environment = MathematicalEnvironment(
        architecture=architecture
    )

    environment.add_model(logical_model)

    return environment


if __name__ == "__main__":
    environment = build_environment()
    state = environment.export()
    pprint.pprint(state)
