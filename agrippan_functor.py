from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, Generic, Iterable, List, Sequence, Tuple, TypeVar


T = TypeVar("T")
Element = int
Vector = Tuple[Element, ...]
Morphism = Callable[[Element], Element]
BinaryOperation = Callable[[Element, Element], Element]


@dataclass(frozen=True)
class DiscreteSpace:
    domain: Vector

    def power(self, rank: int) -> Tuple[Vector, ...]:
        return tuple(product(self.domain, repeat=rank))


@dataclass(frozen=True)
class AbelianObject:
    carrier: Vector
    operation: BinaryOperation

    def compose(self, *elements: Element) -> Element:
        if not elements:
            raise ValueError("empty composition")

        result = elements[0]

        for element in elements[1:]:
            result = self.operation(result, element)

        return result

    def pairwise(self) -> Tuple[Tuple[Element, Element], ...]:
        relations = []

        for i in range(len(self.carrier)):
            for j in range(i + 1, len(self.carrier)):
                relations.append((self.carrier[i], self.carrier[j]))

        return tuple(relations)

    def unary_projection(self) -> Vector:
        return tuple(
            self.operation(x, x)
            for x in self.carrier
        )

    def binary_projection(self) -> Vector:
        return tuple(
            self.operation(a, b)
            for a, b in self.pairwise()
        )

    def total_projection(self) -> Element:
        return self.compose(*self.carrier)


@dataclass(frozen=True)
class Functor(Generic[T]):
    source: str
    target: str
    transform: Morphism

    def map(self, value: T) -> T:
        return self.transform(value)


@dataclass(frozen=True)
class InferentialLayer:
    unary: Vector
    binary: Vector
    total: Element

    def fixed(self) -> bool:
        return (
            len(set(self.unary)) == 1
            and len(set(self.binary)) == 1
            and self.total == 0
        )

    def circular(self) -> bool:
        return (
            len(set(self.binary)) == 1
            and len(set(self.unary)) > 1
        )

    def regress(self) -> bool:
        return (
            len(set(self.binary)) > 1
            and len(set(self.unary)) > 1
        )

    def underdetermined(self) -> bool:
        return not (
            self.fixed()
            or self.circular()
            or self.regress()
        )


class AbelianFunctorSystem:
    def __init__(
        self,
        domain: Sequence[Element],
        operation: BinaryOperation,
    ) -> None:
        self.space = DiscreteSpace(tuple(domain))
        self.operation = operation

        self.identity = Functor(
            source="D",
            target="D",
            transform=lambda x: x,
        )

    def object(
        self,
        values: Sequence[Element],
    ) -> AbelianObject:
        return AbelianObject(
            carrier=tuple(values),
            operation=self.operation,
        )

    def categorical_tensor(
        self,
        rank: int,
    ) -> Tuple[Vector, ...]:
        return self.space.power(rank)

    def inferential_layer(
        self,
        values: Sequence[Element],
    ) -> InferentialLayer:
        obj = self.object(values)

        return InferentialLayer(
            unary=obj.unary_projection(),
            binary=obj.binary_projection(),
            total=obj.total_projection(),
        )

    def agrippan_state(
        self,
        values: Sequence[Element],
    ) -> str:
        layer = self.inferential_layer(values)

        if layer.fixed():
            return "axiomatic_closure"

        if layer.circular():
            return "circular_justification"

        if layer.regress():
            return "infinite_regress"

        return "underdetermined"

    def section(
        self,
        values: Sequence[Element],
    ) -> Dict[str, object]:
        obj = self.object(values)

        return {
            "carrier": obj.carrier,
            "unary": obj.unary_projection(),
            "binary": obj.binary_projection(),
            "total": obj.total_projection(),
            "agrippan_state": self.agrippan_state(values),
        }


def modular_addition(
    modulus: int,
) -> BinaryOperation:
    return lambda a, b: (a + b) % modulus


def relation_matrix(
    system: AbelianFunctorSystem,
    tensors: Iterable[Vector],
) -> List[Dict[str, object]]:
    return [
        system.section(vector)
        for vector in tensors
    ]


def canonical_model() -> Dict[str, object]:
    domain = (0, 1, 2)

    system = AbelianFunctorSystem(
        domain=domain,
        operation=modular_addition(3),
    )

    tensors = system.categorical_tensor(rank=3)

    return {
        "domain": domain,
        "tensor_rank": 3,
        "tensor_space": tensors,
        "relations": relation_matrix(system, tensors),
    }


def run_simulation():
    print("--- Running Agrippan Functor Simulation ---")
    start_time = time.time()

    model = canonical_model()
    relations = model["relations"]

    for relation in relations[:5]:
        print(relation)

    if len(relations) > 5:
        print(f"... and {len(relations) - 5} more relations.")

    duration = time.time() - start_time
    print(f"Simulation took: {duration:.4f}s")
    print("Simulation Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agrippan Functor System")
    parser.add_argument("--simulate", action="store_true", help="Run simulation")
    args = parser.parse_args()

    if args.simulate or not any(vars(args).values()):
        run_simulation()
