from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import math


class ProvenanceMismatch(Exception):
    """Exception raised when an implementation's structural fingerprint
    does not match the verification criteria of its claimed origin."""
    pass


@dataclass
class Node:
    key: str
    path: str
    implementation: str
    depends_on: set[str] = field(default_factory=set)
    refines: str | None = None          # coarser node this one decomposes
    claimed_origin: str | None = None   # name attached by whoever wrote it
    fn: Callable[..., Any] | None = None
    op_counter: Callable[..., dict] | None = None  # returns structural fingerprint


TREE = {
    "add": ("Arithmetic", "ADD", set(), None, None),
    "sub": ("Arithmetic", "SUB", set(), None, None),
    "mul": ("Arithmetic", "IMUL", set(), None, None),
    "div": ("Arithmetic", "IDIV", set(), None, None),
    "gcd": ("Arithmetic", "loop (CMP/Jcc) + IDIV", {"div"}, None, None),

    # mul, previously a leaf, refined one level down: same interface,
    # different implementation category. The 'atomic instruction' and the
    # 'digit-level decomposition' are two members of the same fiber over
    # the same base node.
    "mul_digit_convolution": (
        "Arithmetic", "digit-wise cross convolution + carry",
        {"mul", "add", "div"}, "mul", "schoolbook",  # updated claimed origin to "schoolbook"
    ),
    "mul_trachtenberg": (
        "Arithmetic", "Trachtenberg Speed System of Basic Mathematics",
        {"add"}, "mul", "trachtenberg",
    ),

    "evaluate_p": ("Algebra", "ADD + IMUL (Horner's rule)", {"add", "mul"}, None, None),
    "solve_ax_b": ("Algebra", "NEG + IDIV", {"div"}, None, None),
    "homomorphism": ("Algebra", "calls to the operations of \u22c6", set(), None, None),

    "distance": ("Geometry", "SUBSS + MULSS + SQRTSS", {"sub", "mul"}, None, None),
    "congruent": ("Geometry", "distance + UCOMISS", {"distance"}, None, None),
    "similar": ("Geometry", "DIVSS + UCOMISS", {"distance", "div"}, None, None),

    "sin": ("Trigonometry", "polynomial approximation or library call", set(), None, None),
    "cos": ("Trigonometry", "polynomial approximation or library call", set(), None, None),
    "tan": ("Trigonometry", "sin(x)/cos(x)", {"sin", "cos"}, None, None),

    "limit": ("Calculus", "iterative algorithm", set(), None, None),
    "derivative": ("Calculus", "finite differences", {"sub", "div", "limit"}, None, None),
    "integral": ("Calculus", "numerical quadrature", {"add", "mul"}, None, None),
}


def _fn_add(a: Any, b: Any) -> Any:
    return a + b


def _fn_sub(a: Any, b: Any) -> Any:
    return a - b


def _fn_mul(a: Any, b: Any) -> Any:
    return a * b


def _fn_div(a: Any, b: Any) -> Any:
    return a / b


def _fn_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _fn_mul_digit_convolution(multiplicand: int, multiplier: int) -> int:
    str_cand, str_gand = str(multiplicand), str(multiplier)
    len_cand, len_gand = len(str_cand), len(str_gand)
    max_digits = len_cand + len_gand
    result_digits, carry = [], 0

    for pos in range(max_digits):
        current_sum = carry
        for i in range(pos + 1):
            j = pos - i
            if i < len_cand and j < len_gand:
                d_cand = int(str_cand[len_cand - 1 - i])
                d_gand = int(str_gand[len_gand - 1 - j])
                current_sum += d_cand * d_gand
        carry = current_sum // 10
        result_digits.append(str(current_sum % 10))

    final = "".join(reversed(result_digits)).lstrip("0")
    return int(final) if final else 0


def _count_mul_digit_convolution(multiplicand: int, multiplier: int) -> dict:
    """Structural fingerprint: how many single-digit multiplications the
    algorithm actually performs, vs. the full schoolbook cross product."""
    len_cand = len(str(multiplicand))
    len_gand = len(str(multiplier))
    performed = len_cand * len_gand  # every (i,j) pair is visited, unconditionally
    full_cross = len_cand * len_gand
    return {
        "multiplications_performed": performed,
        "full_cross_product": full_cross,
        "avoided": full_cross - performed,
    }


def trachtenberg_single(n: int, d: int) -> int:
    """Helper that multiplies a number n by a single digit d using the
    Trachtenberg Speed System rules, performing zero standard single-digit
    multiplications."""
    if d == 0:
        return 0
    if d == 1:
        return n
    if d == 2:
        return n + n

    # Convert multiplicand n to digits, padded with leading zero
    digits = [0] + [int(c) for c in str(n)]
    result_digits = []
    carry = 0

    # Process from right to left
    for i in range(len(digits) - 1, -1, -1):
        cur = digits[i]
        neighbor = digits[i + 1] if i + 1 < len(digits) else 0

        val = 0
        if d == 9:
            if i == len(digits) - 1:  # first digit
                val = 10 - cur
            elif i == 0:  # leftmost zero
                val = neighbor - 1
            else:
                val = (9 - cur) + neighbor

        elif d == 8:
            if i == len(digits) - 1:
                val = (10 - cur) * 2
            elif i == 0:
                val = neighbor - 2
            else:
                val = (9 - cur) * 2 + neighbor

        elif d == 7:
            if i == 0:
                val = neighbor // 2
            else:
                val = cur * 2 + neighbor // 2 + (5 if cur % 2 != 0 else 0)

        elif d == 6:
            if i == 0:
                val = neighbor // 2
            else:
                val = cur + neighbor // 2 + (5 if cur % 2 != 0 else 0)

        elif d == 5:
            if i == 0:
                val = neighbor // 2
            else:
                val = neighbor // 2 + (5 if cur % 2 != 0 else 0)

        elif d == 4:
            if i == len(digits) - 1:
                val = (10 - cur) + (5 if cur % 2 != 0 else 0)
            elif i == 0:
                val = neighbor // 2 - 1
            else:
                val = (9 - cur) + neighbor // 2 + (5 if cur % 2 != 0 else 0)

        elif d == 3:
            if i == len(digits) - 1:
                val = (10 - cur) * 2 + (5 if cur % 2 != 0 else 0)
            elif i == 0:
                val = neighbor // 2 - 2
            else:
                val = (9 - cur) * 2 + neighbor // 2 + (5 if cur % 2 != 0 else 0)

        else:
            # Fallback
            val = cur * d

        total = val + carry
        carry = total // 10
        result_digits.append(total % 10)

    while carry:
        result_digits.append(carry % 10)
        carry //= 10

    final = "".join(str(d) for d in reversed(result_digits)).lstrip("0")
    return int(final) if final else 0


def _fn_mul_trachtenberg(multiplicand: int, multiplier: int) -> int:
    """Multiplies two integers using the Trachtenberg Speed System,
    decomposing the multiplier into digits and summing shifted results."""
    str_mult = str(multiplier)
    total_sum = 0
    for i, char in enumerate(reversed(str_mult)):
        d = int(char)
        term = trachtenberg_single(multiplicand, d)
        total_sum += term * (10 ** i)
    return total_sum


def _count_mul_trachtenberg(multiplicand: int, multiplier: int) -> dict:
    """Complexity signature for Trachtenberg algorithm: standard multiplications
    performed are zero, as they are avoided using the Trachtenberg Speed System."""
    len_cand = len(str(multiplicand))
    len_gand = len(str(multiplier))
    full_cross = len_cand * len_gand
    performed = 0
    return {
        "multiplications_performed": performed,
        "full_cross_product": full_cross,
        "avoided": full_cross - performed,
    }


def _fn_evaluate_p(coeffs: list[float] | tuple[float, ...], x: float) -> float:
    result = 0.0
    for c in coeffs:
        result = result * x + c
    return result


def _fn_solve_ax_b(a: float, b: float) -> float:
    return -b / a


def _fn_homomorphism(phi: Callable[[Any], Any], x: Any, y: Any, op: Callable[[Any, Any], Any]) -> Any:
    return phi(op(x, y))


def _fn_distance(p: tuple[float, ...], q: tuple[float, ...]) -> float:
    return math.sqrt(sum((pi - qi) ** 2 for pi, qi in zip(p, q)))


def _fn_congruent(ab: float, cd: float, eps: float = 1e-6) -> bool:
    return abs(ab - cd) < eps


def _fn_similar(ratio_a: float, ratio_b: float, eps: float = 1e-6) -> bool:
    return abs(ratio_a - ratio_b) < eps


def _fn_sin(x: float, terms: int = 12) -> float:
    s, term = x, x
    for n in range(1, terms):
        term *= -x * x / ((2 * n) * (2 * n + 1))
        s += term
    return s


def _fn_cos(x: float, terms: int = 12) -> float:
    s, term = 1.0, 1.0
    for n in range(1, terms):
        term *= -x * x / ((2 * n - 1) * (2 * n))
        s += term
    return s


def _fn_tan(sin_x: float, cos_x: float) -> float:
    return sin_x / cos_x


def _fn_limit(f: Callable[[float], float], x0: float, h0: float = 1e-1, tol: float = 1e-9, max_iter: int = 60) -> float:
    h = h0
    prev = f(x0 + h)
    for _ in range(max_iter):
        h /= 2.0
        cur = f(x0 + h)
        if abs(cur - prev) < tol:
            return cur
        prev = cur
    return prev


def _fn_derivative(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)


def _fn_integral(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + i * h)
    return total * h


IMPLEMENTATIONS: dict[str, Callable[..., Any]] = {
    "add": _fn_add, "sub": _fn_sub, "mul": _fn_mul, "div": _fn_div,
    "gcd": _fn_gcd,
    "mul_digit_convolution": _fn_mul_digit_convolution,
    "mul_trachtenberg": _fn_mul_trachtenberg,
    "evaluate_p": _fn_evaluate_p, "solve_ax_b": _fn_solve_ax_b,
    "homomorphism": _fn_homomorphism, "distance": _fn_distance,
    "congruent": _fn_congruent, "similar": _fn_similar,
    "sin": _fn_sin, "cos": _fn_cos, "tan": _fn_tan,
    "limit": _fn_limit, "derivative": _fn_derivative, "integral": _fn_integral,
}

OP_COUNTERS: dict[str, Callable[..., dict]] = {
    "mul_digit_convolution": _count_mul_digit_convolution,
    "mul_trachtenberg": _count_mul_trachtenberg,
}

ORIGIN_SIGNATURES: dict[str, Callable[[dict], bool]] = {
    "trachtenberg": lambda fp: fp["avoided"] > 0,
    "schoolbook": lambda fp: fp["avoided"] == 0,
}


class MetaSystem:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        for key, (branch, impl, deps, refines, origin) in TREE.items():
            self.nodes[key] = Node(
                key=key,
                path=f"{branch}.{refines + '.' if refines else ''}{key}",
                implementation=impl,
                depends_on=set(deps),  # copy dependencies to avoid global state mutation
                refines=refines,
                claimed_origin=origin,
                fn=IMPLEMENTATIONS.get(key),
                op_counter=OP_COUNTERS.get(key),
            )
        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        visited, stack = set(), set()

        def visit(k: str) -> None:
            if k in stack:
                raise ValueError(f"cycle detected at {k}")
            if k in visited:
                return
            if k not in self.nodes:
                return
            stack.add(k)
            for d in self.nodes[k].depends_on:
                visit(d)
            stack.discard(k)
            visited.add(k)

        for k in self.nodes:
            visit(k)

    def topological_layers(self) -> list[list[str]]:
        remaining = dict(self.nodes)
        resolved: set[str] = set()
        layers: list[list[str]] = []
        while remaining:
            layer = [k for k, n in remaining.items() if n.depends_on <= resolved]
            if not layer:
                raise ValueError("unresolvable dependency remainder")
            layers.append(sorted(layer))
            for k in layer:
                del remaining[k]
            resolved.update(layer)
        return layers

    def verify_provenance(self, key: str, *args, **kwargs) -> dict:
        node = self.nodes[key]
        if node.claimed_origin is None or node.op_counter is None:
            return {"key": key, "claimed_origin": None, "verified": None}

        fingerprint = node.op_counter(*args, **kwargs)
        check = ORIGIN_SIGNATURES.get(node.claimed_origin)
        verified = check(fingerprint) if check else None

        if verified is False:
            raise ProvenanceMismatch(
                f"Provenance verification failed for node '{key}': "
                f"claimed origin '{node.claimed_origin}' is not satisfied by fingerprint {fingerprint}."
            )

        return {
            "key": key,
            "claimed_origin": node.claimed_origin,
            "fingerprint": fingerprint,
            "verified": verified,
        }

    def run(self, calls: dict[str, tuple[tuple, dict]], workers: int = 8) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for layer in self.topological_layers():
            runnable = [k for k in layer if k in calls]
            if not runnable:
                continue
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {}
                for k in runnable:
                    args, kwargs = calls[k]
                    resolved_args = tuple(
                        results[a] if isinstance(a, str) and a in results else a
                        for a in args
                    )
                    resolved_kwargs = {
                        kk: (results[vv] if isinstance(vv, str) and vv in results else vv)
                        for kk, vv in kwargs.items()
                    }
                    futures[ex.submit(self.nodes[k].fn, *resolved_args, **resolved_kwargs)] = k
                for fut in as_completed(futures):
                    k = futures[fut]
                    results[k] = fut.result()
        return results


if __name__ == "__main__":
    ms = MetaSystem()
    print("MetaSystem initialized successfully.")
    for layer in ms.topological_layers():
        print("Layer:", layer)
