#!/usr/bin/env python3
"""Identities of [±a±b] x [±a±b]: matrix, harmonic series, abelian groups,
summary, Huffman indexing and Hamming relations."""

from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict
from fractions import Fraction
from functools import reduce

# ---------------------------------------------------------------------------
# 1. Binomials and the 4x4 matrix of products
# ---------------------------------------------------------------------------
# A binomial c1*a + c2*b is stored as (c1, c2).
# A quadratic form x*a^2 + y*ab + z*b^2 is stored as (x, y, z).
# Bit encoding: 0 -> "+", 1 -> "-" (sign of the a-term, sign of the b-term).

BINOMIAL_BITS = {"u": (0, 0), "v": (0, 1), "-v": (1, 0), "-u": (1, 1)}
ORDER = ["u", "v", "-v", "-u"]  # u = a+b, v = a-b

FORM_NAMES = {
    (1, 2, 1): "A",    # (a+b)^2 = a^2 + 2ab + b^2
    (-1, -2, -1): "-A",
    (1, -2, 1): "E",   # (a-b)^2 = a^2 - 2ab + b^2
    (-1, 2, -1): "-E",
    (1, 0, -1): "B",   # (a+b)(a-b) = a^2 - b^2
    (-1, 0, 1): "-B",
}
FORM_OF = {name: form for form, name in FORM_NAMES.items()}
IDENTITY_ORDER = ["B", "-B", "A", "-A", "E", "-E"]


def binomial_from_bits(alpha: int, beta: int) -> tuple[int, int]:
    return ((-1) ** alpha, (-1) ** beta)


def multiply(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int, int]:
    (c1, c2), (d1, d2) = p, q
    return (c1 * d1, c1 * d2 + c2 * d1, c2 * d2)


def identity_of_cell(cell: str) -> str:
    a1, b1, a2, b2 = (int(ch) for ch in cell)
    return FORM_NAMES[multiply(binomial_from_bits(a1, b1), binomial_from_bits(a2, b2))]


CELLS = ["".join(map(str, bits)) for bits in itertools.product((0, 1), repeat=4)]
CELL_IDENTITY = {c: identity_of_cell(c) for c in CELLS}

MATRIX = [
    [
        FORM_NAMES[multiply(binomial_from_bits(*BINOMIAL_BITS[r]),
                            binomial_from_bits(*BINOMIAL_BITS[c]))]
        for c in ORDER
    ]
    for r in ORDER
]

FIBRES: dict[str, list[str]] = defaultdict(list)
for _cell in CELLS:
    FIBRES[CELL_IDENTITY[_cell]].append(_cell)

EXPECTED_MATRIX = [
    ["A", "B", "-B", "-A"],
    ["B", "E", "-E", "-B"],
    ["-B", "-E", "E", "B"],
    ["-A", "-B", "B", "A"],
]
EXPECTED_MULTIPLICITY = {"A": 2, "-A": 2, "E": 2, "-E": 2, "B": 4, "-B": 4}


def check_matrix() -> None:
    assert MATRIX == EXPECTED_MATRIX
    assert all(MATRIX[i][j] == MATRIX[j][i] for i in range(4) for j in range(4))
    assert {k: len(v) for k, v in FIBRES.items()} == EXPECTED_MULTIPLICITY
    # Rank one: M = g g^T with g = (u, v, -v, -u); checked at a numeric point.
    a, b = 0.7, 0.2
    u, v = a + b, a - b
    g = [u, v, -v, -u]
    for i in range(4):
        for j in range(4):
            assert math.isclose(value(MATRIX[i][j], a, b), g[i] * g[j], abs_tol=1e-12)


def value(name: str, a: float, b: float) -> float:
    x, y, z = FORM_OF[name]
    return x * a * a + y * a * b + z * b * b


# ---------------------------------------------------------------------------
# 2. Simple and alternating harmonic series over the identities
# ---------------------------------------------------------------------------
# Simple:      H(x)   = sum x^n / n            = -ln(1 - x),  |x| < 1
# Alternating: Alt(x) = sum (-1)^(n+1) x^n / n =  ln(1 + x),  |x| <= 1, x != -1


def harmonic_simple(x: float, terms: int = 400) -> float:
    return sum(x ** n / n for n in range(1, terms + 1))


def harmonic_alternating(x: float, terms: int = 400) -> float:
    return sum((-1) ** (n + 1) * x ** n / n for n in range(1, terms + 1))


def series_over_matrix(a: float, b: float, terms: int = 400) -> dict[str, float]:
    cells = [value(MATRIX[i][j], a, b) for i in range(4) for j in range(4)]
    total_simple = sum(harmonic_simple(x, terms) for x in cells)
    total_alt = sum(harmonic_alternating(x, terms) for x in cells)
    A, E, B = value("A", a, b), value("E", a, b), value("B", a, b)
    log_p = 2 * math.log(1 - A * A) + 2 * math.log(1 - E * E) + 4 * math.log(1 - B * B)
    return {
        "total_simple": total_simple,
        "total_alternating": total_alt,
        "minus_ln_P": -log_p,
        "ln_P": log_p,
    }


def check_series(a: float = 0.3, b: float = 0.2) -> dict[str, float]:
    for name in FORM_OF:
        x = value(name, a, b)
        assert abs(x) < 1
        assert math.isclose(harmonic_simple(x), -math.log(1 - x), abs_tol=1e-9)
        assert math.isclose(harmonic_alternating(x), math.log(1 + x), abs_tol=1e-9)
        # The sign of the identity switches between the two series.
        assert math.isclose(harmonic_alternating(x), -harmonic_simple(-x), abs_tol=1e-9)
    res = series_over_matrix(a, b)
    assert math.isclose(res["total_simple"], res["minus_ln_P"], abs_tol=1e-9)
    assert math.isclose(res["total_alternating"], res["ln_P"], abs_tol=1e-9)
    assert math.isclose(res["total_alternating"], -res["total_simple"], abs_tol=1e-9)
    return res


# ---------------------------------------------------------------------------
# 3. Abelian group structure
# ---------------------------------------------------------------------------
# 3a. Multiplicative group G = <-1> x <u> x <v>, isomorphic to C2 x Z^2.
#     An element is (eps, i, j) standing for eps * u^i * v^j.

GElem = tuple[int, int, int]


def gmul(x: GElem, y: GElem) -> GElem:
    return (x[0] * y[0], x[1] + y[1], x[2] + y[2])


def ginv(x: GElem) -> GElem:
    return (x[0], -x[1], -x[2])


def gpow(x: GElem, n: int) -> GElem:
    result: GElem = (1, 0, 0)
    base = x if n >= 0 else ginv(x)
    for _ in range(abs(n)):
        result = gmul(result, base)
    return result


G_BINOMIAL: dict[str, GElem] = {
    "u": (1, 1, 0),
    "v": (1, 0, 1),
    "-v": (-1, 0, 1),
    "-u": (-1, 1, 0),
}
G_IDENTITY: dict[str, GElem] = {
    "A": (1, 2, 0), "-A": (-1, 2, 0),
    "B": (1, 1, 1), "-B": (-1, 1, 1),
    "E": (1, 0, 2), "-E": (-1, 0, 2),
}


def check_multiplicative_group() -> None:
    for r in ORDER:
        for c in ORDER:
            assert gmul(G_BINOMIAL[r], G_BINOMIAL[c]) == G_IDENTITY[MATRIX[ORDER.index(r)][ORDER.index(c)]]
    A, B, E = G_IDENTITY["A"], G_IDENTITY["B"], G_IDENTITY["E"]
    assert gmul(A, E) == gmul(B, B)                      # A * E = B^2
    assert E == gmul(gpow(B, 2), ginv(A))                # E = B^2 * A^-1
    # Hence A*E^2 and B^2*E are one and the same element (exponent (2, 4)).
    assert gmul(A, gpow(E, 2)) == gmul(gpow(B, 2), E) == (1, 2, 4)
    # Index of the even-degree sublattice {(i, j): i + j even} in Z^2.
    assert abs(2 * 1 - 0 * 1) == 2                       # basis (2,0), (1,1)
    # The binomial lattice <(1,1), (1,-1)> has index |det| = 2 in Z^2.
    assert abs(1 * (-1) - 1 * 1) == 2


# 3b. Additive lattice of the identities inside Z^3 (coefficients of a^2, ab, b^2).

def det3(m: list[list[int]]) -> int:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def smith_invariants_3x3(m: list[list[int]]) -> tuple[int, int, int]:
    d1 = reduce(math.gcd, (abs(x) for row in m for x in row))
    minors = []
    for rows in itertools.combinations(range(3), 2):
        for cols in itertools.combinations(range(3), 2):
            r0, r1 = rows
            c0, c1 = cols
            minors.append(abs(m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]))
    d2 = reduce(math.gcd, minors)
    d3 = abs(det3(m))
    return (d1, d2 // d1, d3 // d2)


LATTICE_BASIS = [list(FORM_OF["A"]), list(FORM_OF["E"]), list(FORM_OF["B"])]


def check_additive_lattice() -> None:
    assert abs(det3(LATTICE_BASIS)) == 8                 # index of L in Z^3
    assert smith_invariants_3x3(LATTICE_BASIS) == (1, 2, 4)  # Z^3 / L = Z/2 + Z/4


# 3c. Bit-level homomorphisms (Z/2)^4 -> (Z/2)^2.

def bits(cell: str) -> tuple[int, int, int, int]:
    return tuple(int(ch) for ch in cell)  # type: ignore[return-value]


def syndrome(cell: str) -> tuple[int, int]:
    a1, b1, a2, b2 = bits(cell)
    return (a1 ^ a2, b1 ^ b2)  # (a-premises disagree?, b-premises disagree?)


def xor_cells(c1: str, c2: str) -> str:
    return "".join(str(int(x) ^ int(y)) for x, y in zip(c1, c2))


def check_bit_homomorphisms() -> None:
    for c1, c2 in itertools.product(CELLS, repeat=2):
        s1, s2, s12 = syndrome(c1), syndrome(c2), syndrome(xor_cells(c1, c2))
        assert s12 == (s1[0] ^ s2[0], s1[1] ^ s2[1])
    kernel = sorted(c for c in CELLS if syndrome(c) == (0, 0))
    assert kernel == ["0000", "0101", "1010", "1111"]
    expected_class = {
        (0, 0): {"A", "E"},
        (1, 1): {"-A", "-E"},
        (0, 1): {"B"},
        (1, 0): {"-B"},
    }
    # Syndrome (a-premises disagree, b-premises disagree):
    # (0,0) -> A/E, (1,1) -> -A/-E, (0,1) -> B, (1,0) -> -B.
    for cell in CELLS:
        assert CELL_IDENTITY[cell] in expected_class[syndrome(cell)]
        a1, b1, a2, b2 = bits(cell)
        weight_odd = sum(bits(cell)) % 2 == 1
        assert weight_odd == (CELL_IDENTITY[cell] in {"B", "-B"})
        if CELL_IDENTITY[cell].lstrip("-") in {"A", "E"}:
            # alpha1 xor beta1 separates A (0) from E (1)
            assert (a1 ^ b1 == 1) == (CELL_IDENTITY[cell].lstrip("-") == "E")


# 3d. Symmetries: signed permutations of (a, b) form D4 (order 8, non-abelian).

def substitute(form: tuple[int, int, int], m: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """Apply a -> p*a + q*b, b -> r*a + s*b to x*a^2 + y*ab + z*b^2."""
    x, y, z = form
    p, q, r, s = m
    return (
        x * p * p + y * p * r + z * r * r,
        x * 2 * p * q + y * (p * s + q * r) + z * 2 * r * s,
        x * q * q + y * q * s + z * s * s,
    )


D4_MATRICES: list[tuple[int, int, int, int]] = []
for _sa, _sb in itertools.product((1, -1), repeat=2):
    D4_MATRICES.append((_sa, 0, 0, _sb))
    D4_MATRICES.append((0, _sa, _sb, 0))

NAMES = list(FORM_OF)


def action_on_identities(m: tuple[int, int, int, int]) -> tuple[str, ...]:
    return tuple(FORM_NAMES[substitute(FORM_OF[n], m)] for n in NAMES)


def check_symmetries() -> dict[str, object]:
    assert len(D4_MATRICES) == 8
    actions = {m: action_on_identities(m) for m in D4_MATRICES}
    effective = set(actions.values())
    assert len(effective) == 4                           # D4 / <-I> = C2 x C2
    kernel = [m for m, act in actions.items() if act == tuple(NAMES)]
    assert sorted(kernel) == sorted([(1, 0, 0, 1), (-1, 0, 0, -1)])
    sigma = action_on_identities((1, 0, 0, -1))          # b -> -b : A <-> E
    tau = action_on_identities((0, 1, 1, 0))             # a <-> b  : B <-> -B
    mapping_sigma = dict(zip(NAMES, sigma))
    mapping_tau = dict(zip(NAMES, tau))
    assert mapping_sigma["A"] == "E" and mapping_sigma["B"] == "B"
    assert mapping_tau["B"] == "-B" and mapping_tau["A"] == "A"
    return {"effective_group_order": len(effective), "kernel": kernel}


# ---------------------------------------------------------------------------
# 4. Summary of the identities
# ---------------------------------------------------------------------------

EXPANSION = {
    "A": "a^2 + 2ab + b^2",
    "-A": "-a^2 - 2ab - b^2",
    "E": "a^2 - 2ab + b^2",
    "-E": "-a^2 + 2ab - b^2",
    "B": "a^2 - b^2",
    "-B": "b^2 - a^2",
}


def summary_table() -> list[tuple[str, str, int, GElem]]:
    return [
        (name, EXPANSION[name], len(FIBRES[name]), G_IDENTITY[name])
        for name in ["A", "-A", "E", "-E", "B", "-B"]
    ]


# ---------------------------------------------------------------------------
# 5. Huffman indexing
# ---------------------------------------------------------------------------

def huffman_lengths(freqs: dict[str, Fraction]) -> dict[str, int]:
    heap = [(f, i, [s]) for i, (s, f) in enumerate(freqs.items())]
    heapq.heapify(heap)
    lengths = {s: 0 for s in freqs}
    counter = len(heap)
    while len(heap) > 1:
        f1, _, s1 = heapq.heappop(heap)
        f2, _, s2 = heapq.heappop(heap)
        for s in s1 + s2:
            lengths[s] += 1
        heapq.heappush(heap, (f1 + f2, counter, s1 + s2))
        counter += 1
    return lengths


def canonical_codes(lengths: dict[str, int], order: list[str]) -> dict[str, str]:
    symbols = sorted(lengths, key=lambda s: (lengths[s], order.index(s)))
    codes: dict[str, str] = {}
    code, prev = 0, 0
    for s in symbols:
        length = lengths[s]
        code <<= length - prev
        codes[s] = format(code, f"0{length}b")
        code += 1
        prev = length
    return codes


FREQS = {name: Fraction(len(FIBRES[name]), 16) for name in IDENTITY_ORDER}
HUFFMAN_LENGTHS = huffman_lengths(FREQS)
HUFFMAN_CODES = canonical_codes(HUFFMAN_LENGTHS, IDENTITY_ORDER)

# Suffix bits: the position of a cell inside its identity class.
CELL_WORD: dict[str, str] = {}
for _name in IDENTITY_ORDER:
    _cells = sorted(FIBRES[_name])
    _width = int(math.log2(len(_cells)))
    for _idx, _cell in enumerate(_cells):
        CELL_WORD[_cell] = HUFFMAN_CODES[_name] + (format(_idx, f"0{_width}b") if _width else "")
WORD_CELL = {w: c for c, w in CELL_WORD.items()}


def is_prefix_free(codes: dict[str, str]) -> bool:
    words = list(codes.values())
    return not any(w1 != w2 and w2.startswith(w1) for w1 in words for w2 in words)


def check_huffman() -> dict[str, float]:
    assert HUFFMAN_CODES == {
        "B": "00", "-B": "01", "A": "100", "-A": "101", "E": "110", "-E": "111",
    }
    assert is_prefix_free(HUFFMAN_CODES)
    expected_length = float(sum(FREQS[s] * len(HUFFMAN_CODES[s]) for s in FREQS))
    entropy = -sum(float(p) * math.log2(float(p)) for p in FREQS.values())
    assert math.isclose(expected_length, 2.5) and math.isclose(entropy, 2.5)
    # code length + log2(multiplicity) = 4 for every identity
    for name in FREQS:
        assert len(HUFFMAN_CODES[name]) + int(math.log2(len(FIBRES[name]))) == 4
    # prefix + suffix is a lossless 4-bit address of the original cell
    assert all(len(w) == 4 for w in CELL_WORD.values())
    assert len(WORD_CELL) == 16 and is_prefix_free(CELL_WORD)
    # last bit of the identity code is its sign; the prefix is the exponent class
    for name, code in HUFFMAN_CODES.items():
        assert code[-1] == ("1" if name.startswith("-") else "0")
    return {"expected_length": expected_length, "entropy": entropy}


# ---------------------------------------------------------------------------
# 6. Hamming relations
# ---------------------------------------------------------------------------
# Layout of a Hamming(7,4) word: p1 p2 d1 p3 d2 d3 d4,
# with data bits (d1, d2, d3, d4) = (alpha1, beta1, alpha2, beta2).

def hamming74_encode(d: tuple[int, int, int, int]) -> tuple[int, ...]:
    d1, d2, d3, d4 = d
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return (p1, p2, d1, p3, d2, d3, d4)


def hamming74_decode(word: tuple[int, ...]) -> tuple[tuple[int, int, int, int], int]:
    w = list(word)
    p1, p2, d1, p3, d2, d3, d4 = w
    s1 = p1 ^ d1 ^ d2 ^ d4
    s2 = p2 ^ d1 ^ d3 ^ d4
    s3 = p3 ^ d2 ^ d3 ^ d4
    position = s1 + 2 * s2 + 4 * s3
    if position:
        w[position - 1] ^= 1
    return (w[2], w[4], w[5], w[6]), position


def hamming_distance(x: tuple[int, ...] | str, y: tuple[int, ...] | str) -> int:
    return sum(p != q for p, q in zip(x, y))


HAMMING_WORD = {c: hamming74_encode(bits(c)) for c in CELLS}


def word_string(word: tuple[int, ...]) -> str:
    return "".join(map(str, word))


def class_distance(x: str, y: str) -> int:
    return min(hamming_distance(c1, c2) for c1 in FIBRES[x] for c2 in FIBRES[y])


def check_hamming() -> dict[str, object]:
    codewords = list(HAMMING_WORD.values())
    # minimum distance 3
    assert min(hamming_distance(x, y) for x, y in itertools.combinations(codewords, 2)) == 3
    # perfect: every 7-bit word lies within distance 1 of exactly one codeword
    for word in itertools.product((0, 1), repeat=7):
        near = [cw for cw in codewords if hamming_distance(word, cw) <= 1]
        assert len(near) == 1
    # closed under complement (all-ones is a codeword)
    code_set = set(codewords)
    assert all(tuple(1 - x for x in cw) in code_set for cw in codewords)
    # single-error correction on every cell and every error position
    for cell, word in HAMMING_WORD.items():
        for pos in range(7):
            corrupted = list(word)
            corrupted[pos] ^= 1
            data, _ = hamming74_decode(tuple(corrupted))
            assert data == bits(cell)
    # complementing a cell leaves its identity unchanged: fibres are unions of {c, ~c}
    for cell in CELLS:
        comp = "".join("1" if ch == "0" else "0" for ch in cell)
        assert CELL_IDENTITY[cell] == CELL_IDENTITY[comp]
    # single-bit flips on cells induce the complete bipartite graph K_{2,4}
    # between the odd-weight classes {B, -B} and the even-weight classes
    # {A, -A, E, -E}: beta-flips keep the sign, alpha-flips reverse it.
    edges = set()
    for cell in CELLS:
        for pos in range(4):
            flipped = cell[:pos] + ("1" if cell[pos] == "0" else "0") + cell[pos + 1:]
            if CELL_IDENTITY[cell] != CELL_IDENTITY[flipped]:
                edges.add(frozenset((CELL_IDENTITY[cell], CELL_IDENTITY[flipped])))
    expected_edges = {
        frozenset((odd, even)) for odd in ("B", "-B") for even in ("A", "-A", "E", "-E")
    }
    assert edges == expected_edges
    # Hamming midpoints of A and E are exactly the two square roots of A*E = B^2.
    neighbours = lambda x: {next(iter(e - {x})) for e in edges if x in e}
    assert neighbours("A") & neighbours("E") == {"B", "-B"}
    assert class_distance("A", "B") == 1 and class_distance("B", "E") == 1
    assert class_distance("A", "E") == 2 and class_distance("A", "-A") == 2
    assert class_distance("A", "-E") == 2 and class_distance("B", "-B") == 2
    return {"bipartite_edges": sorted(tuple(sorted(e)) for e in edges)}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main() -> None:
    check_matrix()
    series = check_series()
    check_multiplicative_group()
    check_additive_lattice()
    check_bit_homomorphisms()
    symmetry = check_symmetries()
    huffman = check_huffman()
    hamming = check_hamming()

    print("Matrix of products (rows and columns: u, v, -v, -u)")
    for name, row in zip(ORDER, MATRIX):
        print(f"{name:>3} | " + " ".join(f"{x:>3}" for x in row))

    print("\nSeries over the whole matrix at a=0.3, b=0.2")
    for key, val in series.items():
        print(f"  {key}: {val:.12f}")

    print("\nAbelian structure")
    print("  G = <-1> x <u> x <v> = C2 x Z^2; identities lie in degree 2: i + j = 2")
    print("  Relation: A * E = B^2, E = B^2 * A^-1, A*E^2 = B^2*E = (+, 2, 4)")
    print("  Additive lattice index in Z^3: 8; Z^3 / L = Z/2 + Z/4")
    print(f"  Effective symmetry group order on values: {symmetry['effective_group_order']}")

    print("\nSummary")
    for name, expansion, count, g in summary_table():
        print(f"  {name:>2}  {expansion:<18} cells={count}  G-element={g}")

    print("\nHuffman codes and 4-bit addresses")
    for name in IDENTITY_ORDER:
        words = ", ".join(f"{c}->{CELL_WORD[c]}" for c in sorted(FIBRES[name]))
        print(f"  {name:>2}: {HUFFMAN_CODES[name]:<3} | {words}")
    print(f"  expected length = {huffman['expected_length']} bits, entropy = {huffman['entropy']} bits")

    print("\nHamming(7,4) words per identity")
    for name in IDENTITY_ORDER:
        words = ", ".join(f"{c}->{word_string(HAMMING_WORD[c])}" for c in sorted(FIBRES[name]))
        print(f"  {name:>2}: {words}")
    print("  K_{2,4} edges:", hamming["bipartite_edges"])


if __name__ == "__main__":
    main()
