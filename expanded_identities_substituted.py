#!/usr/bin/env python3
"""Substitution reading of the expanded matrix: A = (a+b)^2 and B = a^2 - b^2,
as polynomials in a and b. Reuses the polynomial tools of expanded_identities.py."""

from __future__ import annotations

import itertools
import math
from collections import Counter

from expanded_identities import (
    BINOMIAL_BITS,
    CELL_FORM_2,
    FIBRES_1,
    FIBRES_2,
    IRREDUCIBLES,
    Poly,
    harmonic_alternating,
    harmonic_simple,
    poly_add,
    poly_key,
    poly_mul,
    poly_scale,
    smith_diagonal,
    variable,
)

# ---------------------------------------------------------------------------
# 1. Binomials as polynomials in (a, b)
# ---------------------------------------------------------------------------

a_, b_ = variable(0, 2), variable(1, 2)
ONE: Poly = {(0, 0): 1}
u_ = poly_add(a_, b_)                       # a + b
v_ = poly_add(a_, poly_scale(-1, b_))       # a - b
A_ = poly_mul(u_, u_)                       # A = (a+b)^2
B_ = poly_mul(u_, v_)                       # B = a^2 - b^2


def plus(p: Poly, q: Poly) -> Poly:
    return poly_add(p, q)


def minus(p: Poly, q: Poly) -> Poly:
    return poly_add(p, poly_scale(-1, q))


# Prime factors: 2, a+b, a-b, a, b, A+b, A-b, a+B, a-B.
PRIMES: list[tuple[str, Poly]] = [
    ("2", {(0, 0): 2}),
    ("a+b", u_),
    ("a-b", v_),
    ("a", a_),
    ("b", b_),
    ("A+b", plus(A_, b_)),
    ("A-b", minus(A_, b_)),
    ("a+B", plus(a_, B_)),
    ("a-B", minus(a_, B_)),
]

# Exponent vector over PRIMES of each irreducible type (gamma, delta, t).
FACTORIZATION: dict[tuple[int, int, int], list[int]] = {
    (0, 0, 0): [0, 1, 0, 0, 0, 0, 0, 0, 0],   # a+b
    (0, 0, 1): [0, 0, 1, 0, 0, 0, 0, 0, 0],   # a-b
    (1, 0, 0): [0, 0, 0, 0, 0, 1, 0, 0, 0],   # A+b
    (1, 0, 1): [0, 0, 0, 0, 0, 0, 1, 0, 0],   # A-b
    (0, 1, 0): [0, 0, 0, 0, 0, 0, 0, 1, 0],   # a+B
    (0, 1, 1): [0, 0, 0, 0, 0, 0, 0, 0, 1],   # a-B
    (1, 1, 0): [1, 1, 0, 1, 0, 0, 0, 0, 0],   # A+B = 2 a (a+b)
    (1, 1, 1): [1, 1, 0, 0, 1, 0, 0, 0, 0],   # A-B = 2 b (a+b)
}


def from_exponents(sign: int, exps: list[int]) -> Poly:
    result = poly_scale(sign, ONE)
    for (_, prime), e in zip(PRIMES, exps):
        for _ in range(e):
            result = poly_mul(result, prime)
    return result


def binomial_poly_2(bits: tuple[int, int, int, int]) -> Poly:
    g, d, al, be = bits
    first = [a_, A_][g]
    second = [b_, B_][d]
    return plus(poly_scale((-1) ** al, first), poly_scale((-1) ** be, second))


def check_factorization() -> None:
    for bits in BINOMIAL_BITS:
        g, d, al, be = bits
        exps = FACTORIZATION[(g, d, al ^ be)]
        # binomial = (-1)^alpha * (x + (-1)^(alpha xor beta) y)
        assert poly_key(binomial_poly_2(bits)) == poly_key(from_exponents((-1) ** al, exps))
    # Optional irreducibility check when sympy is installed.
    try:
        import sympy as sp
    except ImportError:
        return
    a, b = sp.symbols("a b")
    A, B = (a + b) ** 2, a ** 2 - b ** 2
    for expr in (a + b, a - b, A + b, A - b, a + B, a - B):
        assert len(sp.factor_list(sp.expand(expr))[1]) == 1
        assert sp.factor_list(sp.expand(expr))[1][0][1] == 1
    assert sp.factor(sp.expand(A + B)) == 2 * a * (a + b)
    assert sp.factor(sp.expand(A - B)) == 2 * b * (a + b)


# ---------------------------------------------------------------------------
# 2. Multiplicative structure: ambient group C2 x Z^9 over the primes above
# ---------------------------------------------------------------------------

def check_multiplicative() -> dict[str, object]:
    types = sorted(FACTORIZATION)
    rows = [FACTORIZATION[t] for t in types]
    # The 8 binomial classes span a direct summand of Z^9 (all invariants 1): free of rank 8.
    assert smith_diagonal(rows) == [1] * 8
    # The identities (pair sums) span the even-sum sublattice: index 2 in that summand.
    pair_rows = [[x + y for x, y in zip(rows[i], rows[j])]
                 for i in range(8) for j in range(i, 8)]
    assert sorted(smith_diagonal(pair_rows)) == [1] * 7 + [2]
    assert len(IRREDUCIBLES) == 8
    return {"ambient": "C2 x Z^9", "generated": "C2 x Z^8", "identity_index": 2}


# ---------------------------------------------------------------------------
# 3. Additive lattice in Z^15 (monomials a^i b^j with i + j <= 4)
# ---------------------------------------------------------------------------

MONOMIALS_2 = [(i, j) for i in range(5) for j in range(5) if i + j <= 4]


def check_additive() -> dict[str, object]:
    forms = list(FIBRES_2)
    rows = [[dict(form).get(m, 0) for m in MONOMIALS_2] for form in forms]
    diag = smith_diagonal(rows)
    assert len(MONOMIALS_2) == 15 and len(diag) == 9           # rank 9 (it was 10)
    assert sorted(diag) == [1, 1, 1, 2, 2, 2, 2, 4, 4]
    assert math.prod(diag) == 256
    degrees = {max(sum(m) for m, _ in form) for form in forms}
    assert degrees == {2, 3, 4}
    return {"rank": len(diag), "quotient_invariants": [d for d in diag if d > 1],
            "index": math.prod(diag)}


def check_extra_relation() -> None:
    # A+B = 2a(a+b) and A-B = 2b(a+b) force one new linear relation among identities:
    # (a+b)(A+B) = (a-b)(A+B) + (a+b)(A-B) + (a-b)(A-B)
    lhs = poly_mul(u_, plus(A_, B_))
    rhs = poly_add(
        poly_add(poly_mul(v_, plus(A_, B_)), poly_mul(u_, minus(A_, B_))),
        poly_mul(v_, minus(A_, B_)),
    )
    assert poly_key(lhs) == poly_key(rhs)
    # equivalently: b (A+B) = a (A-B)
    assert poly_key(poly_mul(b_, plus(A_, B_))) == poly_key(poly_mul(a_, minus(A_, B_)))
    # and (A+B)(A-B) = 4ab(a+b)^2
    four_ab = poly_scale(4, poly_mul(a_, b_))
    assert poly_key(poly_mul(plus(A_, B_), minus(A_, B_))) == poly_key(
        poly_mul(four_ab, poly_mul(u_, u_))
    )


# ---------------------------------------------------------------------------
# 4. Harmonic series (simple and alternating) at a numeric point
# ---------------------------------------------------------------------------

def numeric_values(a: float, b: float) -> list[float]:
    A, B = (a + b) ** 2, a * a - b * b
    return [a + b, a - b, A + b, A - b, a + B, a - B, A + B, A - B]


def check_series(a: float = 0.2, b: float = 0.1) -> dict[str, float]:
    p = numeric_values(a, b)
    # cell value = (±p_i)(±p_j); multiplicities from the 256 cells
    values = []
    for bits1, bits2 in itertools.product(BINOMIAL_BITS, repeat=2):
        x = poly_key(poly_mul(binomial_poly_2(bits1), binomial_poly_2(bits2)))
        values.append(sum(c * a ** i * b ** j for (i, j), c in x))
    assert all(abs(x) < 1 for x in values)
    total_simple = sum(harmonic_simple(x) for x in values)
    total_alt = sum(harmonic_alternating(x) for x in values)
    log_p = sum(4 * math.log(1 - (p[i] * p[j]) ** 2) for i, j in itertools.combinations(range(8), 2))
    log_p += sum(2 * math.log(1 - p[i] ** 4) for i in range(8))
    assert math.isclose(total_simple, -log_p, abs_tol=1e-9)
    assert math.isclose(total_alt, log_p, abs_tol=1e-9)
    # the two factors sharing a+b: p7 / p8 = a / b
    assert math.isclose(p[6] / p[7], a / b)
    return {"total_simple": total_simple, "total_alternating": total_alt}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main() -> None:
    assert {frozenset(v) for v in FIBRES_1.values()} == {frozenset(v) for v in FIBRES_2.values()}
    assert Counter(len(v) for v in FIBRES_2.values()) == Counter({4: 56, 2: 16})
    assert len({form for form in CELL_FORM_2.values()}) == 72
    check_factorization()
    multiplicative = check_multiplicative()
    additive = check_additive()
    check_extra_relation()
    series = check_series()

    print("same 72 classes and multiplicities as the independent-symbol reading")
    print("primes:", ", ".join(name for name, _ in PRIMES))
    print("A+B = 2a(a+b), A-B = 2b(a+b); other six binomial types are irreducible")
    print("multiplicative:", multiplicative)
    print("additive lattice:", additive)
    print("extra relation: (a+b)(A+B) = (a-b)(A+B) + (a+b)(A-B) + (a-b)(A-B)")
    print("series at a=0.2, b=0.1:", {k: round(v, 12) for k, v in series.items()})


if __name__ == "__main__":
    main()
