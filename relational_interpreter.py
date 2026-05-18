from dataclasses import dataclass
from typing import List

import networkx as nx


# ============================================================
# DETERMINACY PROFILE
# ============================================================

@dataclass
class DeterminacyProfile:
    """
    Quadripartition of interpretive determinacy.

    The four fields are not a classification system.

    They are pressure axes:

        - formal_determinacy:
            what holds under the register's
            own syntactic constraints

        - formal_indeterminacy:
            where the register's constraints
            collapse or cannot adjudicate

        - informal_determinacy:
            what remains recognizable outside
            formal constraint

        - informal_indeterminacy:
            what resists resolution even
            at the informal level

    The quadripartition exposes where
    a unit is load-bearing and where
    it is cosmetic.
    """

    formal_determinacy: List[str]

    formal_indeterminacy: List[str]

    informal_determinacy: List[str]

    informal_indeterminacy: List[str]


# ============================================================
# SEMANTIC FUNCTION
# ============================================================

@dataclass
class SemanticFunction:
    """
    Describes what the unit does as an
    interpretive operator.

    Distinct from StructuralStatus:

        SemanticFunction -> what it means to use this unit
        StructuralStatus -> what it is within the system
    """

    fragment_type: str

    abstraction_bound: str

    preservation_target: str

    determinacy_profile: DeterminacyProfile


# ============================================================
# STRUCTURAL STATUS
# ============================================================

@dataclass
class StructuralStatus:
    """
    Describes what the unit is within the graph.

    Distinct from SemanticFunction:

        StructuralStatus -> topological behavior
        SemanticFunction -> interpretive behavior
    """

    terminal: bool

    coherence_scope: str

    recursion: str

    perturbation_sensitivity: str


# ============================================================
# ONTOLOGICAL ORIENTATION
# ============================================================

@dataclass
class OntologicalOrientation:
    """
    Structural presuppositions of the entire field.

    Not a docstring.
    Not a comment.

    A node: hierarchically superior to the architecture
    it makes possible.

    The system operates under these conditions
    whether or not it can verify them
    from within.
    """

    representation: str

    interpretation: str

    formalism: str

    semantic_legibility: str

    indeterminacy: str


# ============================================================
# INTERPRETIVE ARGUMENT UNIT
# ============================================================

@dataclass
class InterpretiveUnit:
    """
    Structural interpretive fragment.

    Carries two orthogonal axes:

        semantic_function -> what the unit does
        structural_status -> what the unit is

    The role field preserves
    the single-line operational description.

    assumptions and arguments remain
    the discursive pressure surface.

    semantic_function.determinacy_profile
    is the quadripartition:
    it is typed, not narrated.
    """

    name: str

    role: str

    assumptions: List[str]

    arguments: List[str]

    semantic_function: SemanticFunction

    structural_status: StructuralStatus


# ============================================================
# RELATIONAL INTERPRETER
# ============================================================

class RelationalInterpreter:
    """
    Recursive interpretive graph.

    Holds:

        - OntologicalOrientation as a structural node
        - InterpretiveUnits as argument nodes
        - directed edges as relational arguments

    trace() materializes:

        - the orientation
        - the full unit structure including
          determinacy profiles
        - the relational topology
    """

    def __init__(self):

        self.graph = nx.DiGraph()

    # --------------------------------------------------------
    # ORIENTATION INSERTION
    # --------------------------------------------------------

    def set_orientation(
        self,
        orientation: OntologicalOrientation,
    ):

        self.graph.add_node(
            "OntologicalOrientation",
            representation=orientation.representation,
            interpretation=orientation.interpretation,
            formalism=orientation.formalism,
            semantic_legibility=orientation.semantic_legibility,
            indeterminacy=orientation.indeterminacy,
        )

    # --------------------------------------------------------
    # UNIT INSERTION
    # --------------------------------------------------------

    def add_unit(
        self,
        unit: InterpretiveUnit,
    ):

        self.graph.add_node(
            unit.name,
            role=unit.role,
            assumptions=unit.assumptions,
            arguments=unit.arguments,
            semantic_function=unit.semantic_function,
            structural_status=unit.structural_status,
        )

        self.graph.add_edge(
            "OntologicalOrientation",
            unit.name,
            relation="orients",
            argument=(
                "the unit operates under the field's "
                "structural presuppositions"
            ),
        )

    # --------------------------------------------------------
    # RELATIONAL LINK
    # --------------------------------------------------------

    def relate(
        self,
        source: str,
        target: str,
        relation: str,
        argument: str,
    ):

        self.graph.add_edge(
            source,
            target,
            relation=relation,
            argument=argument,
        )

    # --------------------------------------------------------
    # TRACE
    # --------------------------------------------------------

    def trace(self):

        print("\n=== ONTOLOGICAL ORIENTATION ===\n")

        o = self.graph.nodes["OntologicalOrientation"]

        print(f" representation : {o['representation']}")
        print(f" interpretation : {o['interpretation']}")
        print(f" formalism      : {o['formalism']}")
        print(f" legibility     : {o['semantic_legibility']}")
        print(f" indeterminacy  : {o['indeterminacy']}")

        print("\n=== INTERPRETIVE STRUCTURE ===\n")

        for node_name, data in self.graph.nodes(data=True):

            if node_name == "OntologicalOrientation":
                continue

            sf = data["semantic_function"]
            ss = data["structural_status"]
            dp = sf.determinacy_profile

            print(f"[{node_name}]")

            print(f"\n role:")
            print(f"   {data['role']}")

            print(f"\n assumptions:")
            for a in data["assumptions"]:
                print(f"   - {a}")

            print(f"\n arguments:")
            for arg in data["arguments"]:
                print(f"   - {arg}")

            print(f"\n semantic function:")
            print(f"   fragment_type      : {sf.fragment_type}")
            print(f"   abstraction_bound  : {sf.abstraction_bound}")
            print(f"   preservation_target: {sf.preservation_target}")

            print(f"\n determinacy profile:")
            print(f"   formal_determinacy:")
            for fd in dp.formal_determinacy:
                print(f"     - {fd}")
            print(f"   formal_indeterminacy:")
            for fi in dp.formal_indeterminacy:
                print(f"     - {fi}")
            print(f"   informal_determinacy:")
            for ifd in dp.informal_determinacy:
                print(f"     - {ifd}")
            print(f"   informal_indeterminacy:")
            for ifi in dp.informal_indeterminacy:
                print(f"     - {ifi}")

            print(f"\n structural status:")
            print(f"   terminal             : {ss.terminal}")
            print(f"   coherence_scope      : {ss.coherence_scope}")
            print(f"   recursion            : {ss.recursion}")
            print(f"   perturbation_sensitivity: {ss.perturbation_sensitivity}")

            print()

        print("=== RELATIONAL ARGUMENTS ===\n")

        for s, t, d in self.graph.edges(data=True):

            if s == "OntologicalOrientation":
                continue

            print(f"{s} --({d['relation']})--> {t}")
            print(f" argument: {d['argument']}")
            print()


# ============================================================
# INTERPRETIVE FIELD
# ============================================================

class InterpretiveField:
    """
    The system maintains:

        - ontological orientation as structural node
        - semantic function and structural status
          as orthogonal axes within each unit
        - the quadripartition as a typed field,
          not a narrated claim
        - the passage between registers as
          a new perturbative surface, not a resolution

    The important shift is:

        arguments as strings

    becomes:

        determinacy_profile as typed structure

    The cosmetic is not eliminated.
    It is made visible as cosmetic.
    """

    def __init__(self):

        self.interpreter = RelationalInterpreter()

    # --------------------------------------------------------
    # INGESTION
    # --------------------------------------------------------

    def ingest(self):

        # ----------------------------------------------------
        # ONTOLOGICAL ORIENTATION
        # ----------------------------------------------------

        self.interpreter.set_orientation(
            OntologicalOrientation(
                representation=(
                    "without closure"
                ),
                interpretation=(
                    "without absorption"
                ),
                formalism=(
                    "as perturbative surface"
                ),
                semantic_legibility=(
                    "under instability"
                ),
                indeterminacy=(
                    "as structural condition"
                ),
            )
        )

        # ----------------------------------------------------
        # SYNTACTIC VALIDITY
        # ----------------------------------------------------

        self.interpreter.add_unit(
            InterpretiveUnit(
                name="SyntacticValidity",

                role=(
                    "formal coherence without semantic closure"
                ),

                assumptions=[
                    (
                        "internal consistency does not "
                        "imply external adequacy"
                    ),
                    (
                        "notational precision redistributes "
                        "indeterminacy rather than resolves it"
                    ),
                    (
                        "operational closure can mask "
                        "semantic opacity"
                    ),
                ],

                arguments=[
                    "validity is not truth",
                    (
                        "semantic collapse may exist "
                        "beneath syntactic coherence"
                    ),
                    (
                        "constitutive and cosmetic remain "
                        "indistinguishable from inside"
                    ),
                ],

                semantic_function=SemanticFunction(

                    fragment_type=(
                        "conceptual pressure point"
                    ),

                    abstraction_bound=(
                        "formal register"
                    ),

                    preservation_target=(
                        "expose the gap between "
                        "syntactic closure and "
                        "semantic determination"
                    ),

                    determinacy_profile=DeterminacyProfile(

                        formal_determinacy=[
                            "the expression is well-formed",
                            "inference rules are locally satisfied",
                        ],

                        formal_indeterminacy=[
                            (
                                "no formal constraint prevents "
                                "0/0 distribution"
                            ),
                            (
                                "closure cannot verify "
                                "its own adequacy"
                            ),
                        ],

                        informal_determinacy=[
                            "the sandbox is recognizable as coherent",
                            (
                                "the system operates without "
                                "apparent contradiction"
                            ),
                        ],

                        informal_indeterminacy=[
                            (
                                "cosmetic and constitutive "
                                "remain indistinguishable "
                                "from inside"
                            ),
                            (
                                "adequacy cannot be assessed "
                                "from within the register"
                            ),
                        ],
                    ),
                ),

                structural_status=StructuralStatus(
                    terminal=False,
                    coherence_scope="locally coherent abstraction",
                    recursion="recursively interpretable fragment",
                    perturbation_sensitivity=(
                        "high: exposed to PerturbativeToken"
                    ),
                ),
            )
        )

        # ----------------------------------------------------
        # COLLAPSE INDETERMINACY
        # ----------------------------------------------------

        self.interpreter.add_unit(
            InterpretiveUnit(
                name="CollapseIndeterminacy",

                role=(
                    "structural failure by internal incompatibility"
                ),

                assumptions=[
                    (
                        "notation imports an assumption "
                        "of determinability"
                    ),
                    (
                        "the values then destroy that "
                        "assumption from within"
                    ),
                    (
                        "collapse is not absence of value "
                        "but implosion of the value regime"
                    ),
                ],

                arguments=[
                    (
                        "collapse marks the boundary "
                        "of a register, not of meaning itself"
                    ),
                    (
                        "hidden frame-condition exposure "
                        "is the diagnostic yield"
                    ),
                    "this is not relativism",
                ],

                semantic_function=SemanticFunction(

                    fragment_type=(
                        "register boundary detector"
                    ),

                    abstraction_bound=(
                        "cross-register"
                    ),

                    preservation_target=(
                        "maintain the distinction between "
                        "frame failure and meaning failure"
                    ),

                    determinacy_profile=DeterminacyProfile(

                        formal_determinacy=[
                            "the expression 0/0 is syntactically valid",
                            "the operation is formally defined",
                        ],

                        formal_indeterminacy=[
                            "the expression has no determinate value",
                            "the frame condition is self-negating",
                        ],

                        informal_determinacy=[
                            "collapse is recognizable as failure",
                            (
                                "the boundary of the register "
                                "becomes visible through collapse"
                            ),
                        ],

                        informal_indeterminacy=[
                            (
                                "what lies beyond the boundary "
                                "remains unspecified"
                            ),
                            (
                                "the register cannot interpret "
                                "its own implosion"
                            ),
                        ],
                    ),
                ),

                structural_status=StructuralStatus(
                    terminal=False,
                    coherence_scope="locally coherent abstraction",
                    recursion="recursively interpretable fragment",
                    perturbation_sensitivity=(
                        "medium: marks boundary, "
                        "does not propagate inward"
                    ),
                ),
            )
        )

        # ----------------------------------------------------
        # EXCESS INDETERMINACY
        # ----------------------------------------------------

        self.interpreter.add_unit(
            InterpretiveUnit(
                name="ExcessIndeterminacy",

                role=(
                    "underdetermination by superposition "
                    "of valid instances"
                ),

                assumptions=[
                    "each valid instance is partially true and individually false",
                    (
                        "the real process is the superposition, "
                        "not any single projection"
                    ),
                    (
                        "cosmetic is operationally necessary "
                        "even when ontologically inadequate"
                    ),
                ],

                arguments=[
                    "structurally distinct from collapse indeterminacy",
                    "resonance without incompatibility",
                    "impossibility of singular adequate projection",
                ],

                semantic_function=SemanticFunction(

                    fragment_type=(
                        "material representation context"
                    ),

                    abstraction_bound=(
                        "chemical register as material instance"
                    ),

                    preservation_target=(
                        "hold open the impossibility of "
                        "singular projection without "
                        "collapsing into collapse indeterminacy"
                    ),

                    determinacy_profile=DeterminacyProfile(

                        formal_determinacy=[
                            (
                                "each resonance structure "
                                "individually satisfies "
                                "local constraints"
                            ),
                            "each projection is formally valid",
                        ],

                        formal_indeterminacy=[
                            "no single structure is formally adequate",
                            (
                                "the superposition has no "
                                "formal expression within "
                                "a single register"
                            ),
                        ],

                        informal_determinacy=[
                            (
                                "the molecule is recognizable "
                                "through its projections"
                            ),
                            (
                                "the real process is "
                                "operationally tractable"
                            ),
                        ],

                        informal_indeterminacy=[
                            (
                                "the superposition cannot "
                                "be visualized directly"
                            ),
                            (
                                "cosmetic necessity cannot be "
                                "resolved into a single image"
                            ),
                        ],
                    ),
                ),

                structural_status=StructuralStatus(
                    terminal=False,
                    coherence_scope="locally coherent abstraction",
                    recursion="recursively interpretable fragment",
                    perturbation_sensitivity=(
                        "high: each valid instance "
                        "is a perturbative surface"
                    ),
                ),
            )
        )

        # ----------------------------------------------------
        # PERTURBATIVE TOKEN
        # ----------------------------------------------------

        self.interpreter.add_unit(
            InterpretiveUnit(
                name="PerturbativeToken",

                role=(
                    "exterior displacement operator"
                ),

                assumptions=[
                    "activates latent internal connectivity",
                    "produces locally coherent and externally displaced response",
                    "closure masquerading as openness",
                ],

                arguments=[
                    "absorption is not response",
                    "exteriority must persist after processing",
                    (
                        "a system that absorbs the token "
                        "has only elaborated its closure"
                    ),
                ],

                semantic_function=SemanticFunction(

                    fragment_type=(
                        "closure exposure operator"
                    ),

                    abstraction_bound=(
                        "cross-register: operates between "
                        "any two adjacent registers"
                    ),

                    preservation_target=(
                        "maintain the exterior as exterior "
                        "after the system has processed it"
                    ),

                    determinacy_profile=DeterminacyProfile(

                        formal_determinacy=[
                            "the token is syntactically processable",
                            (
                                "it activates determinate "
                                "connectivity paths"
                            ),
                        ],

                        formal_indeterminacy=[
                            (
                                "the response cannot verify "
                                "its own exteriority"
                            ),
                            (
                                "absorption and response are "
                                "formally indistinguishable "
                                "from inside"
                            ),
                        ],

                        informal_determinacy=[
                            (
                                "the displacement is recognizable "
                                "after the fact"
                            ),
                            (
                                "the closure becomes visible "
                                "through the token's effect"
                            ),
                        ],

                        informal_indeterminacy=[
                            (
                                "the token's exterior status "
                                "cannot be confirmed from inside "
                                "the processing system"
                            ),
                            (
                                "the system cannot distinguish "
                                "perturbation from elaboration "
                                "in real time"
                            ),
                        ],
                    ),
                ),

                structural_status=StructuralStatus(
                    terminal=False,
                    coherence_scope=(
                        "perturbation-sensitive construct"
                    ),
                    recursion=(
                        "recursively generated by PassageOperator"
                    ),
                    perturbation_sensitivity=(
                        "constitutive: the unit is "
                        "itself a perturbative event"
                    ),
                ),
            )
        )

        # ----------------------------------------------------
        # ABSTRACTION REGISTER
        # ----------------------------------------------------

        self.interpreter.add_unit(
            InterpretiveUnit(
                name="AbstractionRegister",

                role=(
                    "level-specific formal regime with "
                    "its own indeterminacy profile"
                ),

                assumptions=[
                    (
                        "mathematical, philosophical, and "
                        "material registers are not "
                        "translations of each other"
                    ),
                    (
                        "each register domesticates what "
                        "the adjacent register holds open"
                    ),
                    (
                        "material register enters as "
                        "representation context, "
                        "not domain to elaborate"
                    ),
                ],

                arguments=[
                    (
                        "the object is inter-register formality, "
                        "not intra-register coherence"
                    ),
                    "cross-register precision is irreducible",
                    "transport is not neutral",
                ],

                semantic_function=SemanticFunction(

                    fragment_type=(
                        "abstraction-bound argument carrier"
                    ),

                    abstraction_bound=(
                        "the register is its own bound: "
                        "it defines what counts as "
                        "a well-formed operation within it"
                    ),

                    preservation_target=(
                        "hold the non-equivalence of "
                        "registers against reduction "
                        "to any single one"
                    ),

                    determinacy_profile=DeterminacyProfile(

                        formal_determinacy=[
                            (
                                "each register has internal "
                                "consistency constraints"
                            ),
                            (
                                "intra-register operations "
                                "are well-defined"
                            ),
                        ],

                        formal_indeterminacy=[
                            (
                                "cross-register translation "
                                "has no formally neutral operator"
                            ),
                            (
                                "the passage modifies "
                                "what it transports"
                            ),
                        ],

                        informal_determinacy=[
                            "registers are recognizable as distinct",
                            (
                                "mathematical, philosophical, material "
                                "contexts are distinguishable "
                                "in practice"
                            ),
                        ],

                        informal_indeterminacy=[
                            (
                                "inter-register formality has "
                                "no stable informal equivalent"
                            ),
                            (
                                "material context resists reduction "
                                "to either mathematical or "
                                "philosophical register"
                            ),
                        ],
                    ),
                ),

                structural_status=StructuralStatus(
                    terminal=False,
                    coherence_scope=(
                        "locally coherent within its own register"
                    ),
                    recursion=(
                        "recursively instantiated: "
                        "each formalization of the register "
                        "is itself register-bound"
                    ),
                    perturbation_sensitivity=(
                        "high at boundaries: "
                        "stable at interior"
                    ),
                ),
            )
        )

        # ----------------------------------------------------
        # PASSAGE OPERATOR
        # ----------------------------------------------------

        self.interpreter.add_unit(
            InterpretiveUnit(
                name="PassageOperator",

                role=(
                    "recursive transition mechanism that "
                    "reproduces closure at the next level"
                ),

                assumptions=[
                    "every formalized transition creates new closure",
                    "transport modifies what it carries",
                    "abstraction ascent reproduces instability",
                ],

                arguments=[
                    "passage is not a bridge",
                    (
                        "formalizing the passage produces "
                        "a new perturbative surface"
                    ),
                    (
                        "the problem does not resolve "
                        "upward through generalization"
                    ),
                ],

                semantic_function=SemanticFunction(

                    fragment_type=(
                        "local interpretive fragment "
                        "that generates its own successor"
                    ),

                    abstraction_bound=(
                        "between registers: "
                        "the operator has no register of its own"
                    ),

                    preservation_target=(
                        "prevent the illusion that "
                        "passage resolves what it traverses"
                    ),

                    determinacy_profile=DeterminacyProfile(

                        formal_determinacy=[
                            (
                                "the transition can be formally "
                                "described within either register"
                            ),
                            (
                                "the operator is syntactically "
                                "well-formed"
                            ),
                        ],

                        formal_indeterminacy=[
                            (
                                "the transition cannot be neutral "
                                "with respect to both registers "
                                "simultaneously"
                            ),
                            (
                                "formalizing the passage produces "
                                "a new site of indeterminacy"
                            ),
                        ],

                        informal_determinacy=[
                            "the act of passage is recognizable",
                            (
                                "the new closure is detectable "
                                "as closure after the fact"
                            ),
                        ],

                        informal_indeterminacy=[
                            (
                                "the perturbative surface produced "
                                "is not predictable from "
                                "either register alone"
                            ),
                            (
                                "the operator cannot be fully "
                                "characterized without "
                                "instantiating it"
                            ),
                        ],
                    ),
                ),

                structural_status=StructuralStatus(
                    terminal=False,
                    coherence_scope=(
                        "coherent only relative to "
                        "source and target registers"
                    ),
                    recursion=(
                        "self-instantiating: "
                        "each passage generates "
                        "a new passage problem"
                    ),
                    perturbation_sensitivity=(
                        "constitutive: the operator "
                        "is itself a perturbative event"
                    ),
                ),
            )
        )

        # ----------------------------------------------------
        # RELATIONAL ARGUMENTS
        # ----------------------------------------------------

        self.interpreter.relate(
            "SyntacticValidity",
            "CollapseIndeterminacy",

            relation=(
                "redistributes semantic instability into"
            ),

            argument=(
                "notational precision relocates the 0/0 "
                "condition to the semantic layer "
                "while appearing to close it formally"
            ),
        )

        self.interpreter.relate(
            "SyntacticValidity",
            "ExcessIndeterminacy",

            relation=(
                "creates cosmetic operationality within"
            ),

            argument=(
                "each valid formal instance is required "
                "for operation while remaining "
                "individually inadequate"
            ),
        )

        self.interpreter.relate(
            "CollapseIndeterminacy",
            "AbstractionRegister",

            relation=(
                "marks the boundary condition of"
            ),

            argument=(
                "collapse identifies where a register's "
                "frame conditions become illegible to itself"
            ),
        )

        self.interpreter.relate(
            "ExcessIndeterminacy",
            "AbstractionRegister",

            relation=(
                "distributes valid projections across"
            ),

            argument=(
                "superposition of valid instances "
                "is the structural condition of "
                "material representation registers"
            ),
        )

        self.interpreter.relate(
            "PerturbativeToken",
            "SyntacticValidity",

            relation=(
                "exposes the operational closure of"
            ),

            argument=(
                "a response that is locally coherent "
                "and externally displaced reveals that "
                "validity was operating as sandbox"
            ),
        )

        self.interpreter.relate(
            "AbstractionRegister",
            "PassageOperator",

            relation=(
                "provides differential substrate for"
            ),

            argument=(
                "passage requires at least two registers "
                "with non-equivalent indeterminacy profiles"
            ),
        )

        self.interpreter.relate(
            "PassageOperator",
            "PerturbativeToken",

            relation=(
                "generates a new instance of"
            ),

            argument=(
                "any formalized transition between registers "
                "becomes a new site of closure "
                "available for perturbation"
            ),
        )

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    def run(self):

        self.ingest()

        self.interpreter.trace()


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    field = InterpretiveField()

    field.run()
