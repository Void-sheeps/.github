#!/usr/bin/env python3
"""Expanded matrix: families [±a±b], [±A±b], [±a±B], [±A±B] (16 binomials)
multiplied against themselves (256 cells).

Reading 1: a, b, A, B are independent symbols.
Reading 2: A = (a+b)^2 and B = a^2 - b^2, polynomials in a and b.
Both readings induce the same partition of the 256 cells."""

from __future__ import annotations

import heapq
import itertools
import math
from collections import Counter, defaultdict
from fractions import Fraction

# ---------------------------------------------------------------------------
# 1. Minimal polynomial arithmetic: {exponent tuple: integer coefficient}
# ---------------------------------------------------------------------------

Poly = dict[tuple[int, ...], int]


def poly_add(p: Poly, q: Poly) -> Poly:
    r: dict[tuple[int, ...], int] = defaultdict(int)
    for k, v in p.items():
        r[k] += v
    for k, v in q.items():
        r[k] += v
    return {k: v for k, v in r.items() if v}


def poly_scale(s: int, p: Poly) -> Poly:
    return {k: s * v for k, v in p.items()}


def poly_mul(p: Poly, q: Poly) -> Poly:
    r: dict[tuple[int, ...], int] = defaultdict(int)
    for k1, v1 in p.items():
        for k2, v2 in q.items():
            r[tuple(x + y for x, y in zip(k1, k2))] += v1 * v2
    return {k: v for k, v in r.items() if v}


def poly_key(p: Poly) -> tuple:
    return tuple(sorted(p.items()))


def variable(index: int, n: int) -> Poly:
    e = [0] * n
    e[index] = 1
    return {tuple(e): 1}


# ---------------------------------------------------------------------------
# 2. Binomials: bits (gamma, delta, alpha, beta)
#    gamma: 0 -> a, 1 -> A      (first variable)
#    delta: 0 -> b, 1 -> B      (second variable)
#    alpha, beta: 0 -> "+", 1 -> "-" (signs of the first and second term)
# ---------------------------------------------------------------------------

Bits4 = tuple[int, int, int, int]
BINOMIAL_BITS: list[Bits4] = list(itertools.product((0, 1), repeat=4))  # type: ignore[arg-type]
CELLS = ["".join(map(str, p)) for p in itertools.product((0, 1), repeat=8)]


def binomial_poly(bits: Bits4, first: list[Poly], second: list[Poly]) -> Poly:
    g, d, al, be = bits
    return poly_add(poly_scale((-1) ** al, first[g]), poly_scale((-1) ** be, second[d]))


def build_cell_forms(first: list[Poly], second: list[Poly]) -> dict[str, tuple]:
    polys = {bits: binomial_poly(bits, first, second) for bits in BINOMIAL_BITS}
    out = {}
    for b1, b2 in itertools.product(BINOMIAL_BITS, repeat=2):
        cell = "".join(map(str, b1 + b2))
        out[cell] = poly_key(poly_mul(polys[b1], polys[b2]))
    return out


# Reading 1: independent symbols, variable order (a, b, A, B).
_a, _b, _A, _B = (variable(i, 4) for i in range(4))
CELL_FORM_1 = build_cell_forms([_a, _A], [_b, _B])

# Reading 2: A = (a+b)^2, B = a^2 - b^2 in the variables (a, b).
_a2, _b2 = variable(0, 2), variable(1, 2)
_u = poly_add(_a2, _b2)
_v = poly_add(_a2, poly_scale(-1, _b2))
CELL_FORM_2 = build_cell_forms([_a2, poly_mul(_u, _u)], [_b2, poly_mul(_u, _v)])


def fibres_of(cell_form: dict[str, tuple]) -> dict[tuple, list[str]]:
    out: dict[tuple, list[str]] = defaultdict(list)
    for cell, form in cell_form.items():
        out[form].append(cell)
    return out


FIBRES_1 = fibres_of(CELL_FORM_1)
FIBRES_2 = fibres_of(CELL_FORM_2)


def check_partitions() -> None:
    assert len(FIBRES_1) == 72
    assert Counter(len(v) for v in FIBRES_1.values()) == Counter({4: 56, 2: 16})
    part1 = {frozenset(v) for v in FIBRES_1.values()}
    part2 = {frozenset(v) for v in FIBRES_2.values()}
    assert part1 == part2  # both readings give the same 72 classes


# ---------------------------------------------------------------------------
# 3. Identities as unordered pairs of irreducibles with a sign
# ---------------------------------------------------------------------------
# Irreducible id (gamma, delta, t) with t = alpha xor beta: x + y (t=0), x - y (t=1).

Irr = tuple[int, int, int]
IRREDUCIBLES: list[Irr] = sorted({(g, d, al ^ be) for g, d, al, be in BINOMIAL_BITS})
IRR_INDEX = {irr: i for i, irr in enumerate(IRREDUCIBLES)}


def irr_name(irr: Irr) -> str:
    g, d, t = irr
    return f"{'A' if g else 'a'}{'-' if t else '+'}{'B' if d else 'b'}"


def irr_of(bits: Bits4) -> Irr:
    g, d, al, be = bits
    return (g, d, al ^ be)


def cell_label(cell: str) -> tuple[tuple[Irr, Irr], int]:
    bits = tuple(int(ch) for ch in cell)
    b1, b2 = bits[:4], bits[4:]
    pair = tuple(sorted((irr_of(b1), irr_of(b2))))  # type: ignore[arg-type]
    return pair, b1[2] ^ b2[2]  # (unordered pair of irreducibles, sign bit)


def label_name(label: tuple[tuple[Irr, Irr], int]) -> str:
    (i, j), eps = label
    return f"{'-' if eps else '+'}({irr_name(i)})({irr_name(j)})"


CELL_LABEL = {c: cell_label(c) for c in CELLS}
LABEL_FIBRES: dict[tuple, list[str]] = defaultdict(list)
for _c, _l in CELL_LABEL.items():
    LABEL_FIBRES[_l].append(_c)


def check_labels() -> None:
    assert len(IRREDUCIBLES) == 8
    assert {frozenset(v) for v in LABEL_FIBRES.values()} == {frozenset(v) for v in FIBRES_1.values()}
    pairs = Counter(len(v) for v in LABEL_FIBRES.values())
    assert pairs == Counter({4: 56, 2: 16})
    # 28 unordered pairs i < j and 8 squares, each with two signs.
    assert len(LABEL_FIBRES) == 2 * (28 + 8)


# ---------------------------------------------------------------------------
# 4. Abelian structure
# ---------------------------------------------------------------------------
# G = <-1> x Z^8 : element (eps, exponent vector over the 8 irreducibles).

GElem = tuple[int, tuple[int, ...]]


def gmul(x: GElem, y: GElem) -> GElem:
    return (x[0] * y[0], tuple(p + q for p, q in zip(x[1], y[1])))


def g_binomial(bits: Bits4) -> GElem:
    e = [0] * 8
    e[IRR_INDEX[irr_of(bits)]] = 1
    return ((-1) ** bits[2], tuple(e))


def g_label(label: tuple[tuple[Irr, Irr], int]) -> GElem:
    (i, j), eps = label
    e = [0] * 8
    e[IRR_INDEX[i]] += 1
    e[IRR_INDEX[j]] += 1
    return ((-1) ** eps, tuple(e))


def smith_diagonal(rows: list[list[int]]) -> list[int]:
    """Nonzero diagonal of the Smith normal form of an integer matrix."""
    M = [list(r) for r in rows]
    m, n = len(M), len(M[0])
    diag: list[int] = []
    for t in range(min(m, n)):
        while True:
            best = None
            for i in range(t, m):
                for j in range(t, n):
                    if M[i][j] and (best is None or abs(M[i][j]) < abs(M[best[0]][best[1]])):
                        best = (i, j)
            if best is None:
                return diag
            i, j = best
            M[t], M[i] = M[i], M[t]
            for r in M:
                r[t], r[j] = r[j], r[t]
            p = M[t][t]
            clean = True
            for i in range(t + 1, m):
                q = M[i][t] // p
                if q:
                    M[i] = [x - q * y for x, y in zip(M[i], M[t])]
                if M[i][t]:
                    clean = False
            for j in range(t + 1, n):
                q = M[t][j] // p
                if q:
                    for r in M:
                        r[j] -= q * r[t]
                if M[t][j]:
                    clean = False
            if not clean:
                continue
            bad = next(
                ((i, j) for i in range(t + 1, m) for j in range(t + 1, n) if M[i][j] % p), None
            )
            if bad:
                M[t] = [x + y for x, y in zip(M[t], M[bad[0]])]
                continue
            break
        diag.append(abs(M[t][t]))
    return diag


MONOMIALS = sorted({k for form in FIBRES_1 for k, _ in form})


def check_abelian() -> dict[str, object]:
    # Each cell is the group product of its two binomials.
    element_of_label: dict[tuple, GElem] = {}
    for cell, label in CELL_LABEL.items():
        bits = tuple(int(ch) for ch in cell)
        prod = gmul(g_binomial(bits[:4]), g_binomial(bits[4:]))  # type: ignore[arg-type]
        assert prod == g_label(label)
        element_of_label[label] = prod
    assert len(set(element_of_label.values())) == 72           # injective on the 72 classes
    assert all(sum(e) == 2 for _, e in element_of_label.values())  # all of degree 2
    # Commutative-monoid relations: (p_i p_j)(p_k p_l) = (p_i p_k)(p_j p_l) = (p_i p_l)(p_j p_k)
    def pp(i: int, j: int) -> GElem:
        e = [0] * 8
        e[i] += 1
        e[j] += 1
        return (1, tuple(e))
    for i, j, k, l in itertools.product(range(8), repeat=4):
        assert gmul(pp(i, j), pp(k, l)) == gmul(pp(i, k), pp(j, l)) == gmul(pp(i, l), pp(j, k))
    # Exponent lattice generated by the identities: even-sum vectors, index 2 in Z^8.
    exp_rows = [list(e) for _, e in {g_label(l)[1]: g_label(l) for l in LABEL_FIBRES}.values()]
    assert sorted(smith_diagonal(exp_rows)) == [1] * 7 + [2]
    # Additive lattice of the identities inside Z^10 (degree-2 monomials of a, b, A, B).
    rows = [[dict(form).get(m, 0) for m in MONOMIALS] for form in FIBRES_1]
    diag = smith_diagonal(rows)
    assert len(MONOMIALS) == 10 and len(diag) == 10
    assert sorted(diag) == [1] * 6 + [2] * 3 + [4]
    assert math.prod(diag) == 32
    return {"quotient_invariants": [d for d in diag if d > 1], "index": math.prod(diag)}


# ---------------------------------------------------------------------------
# 5. Harmonic series (simple and alternating) over the 256 cells
# ---------------------------------------------------------------------------

def harmonic_simple(x: float, terms: int = 400) -> float:
    return sum(x ** n / n for n in range(1, terms + 1))


def harmonic_alternating(x: float, terms: int = 400) -> float:
    return sum((-1) ** (n + 1) * x ** n / n for n in range(1, terms + 1))


def linear_value(irr: Irr, point: dict[str, float]) -> float:
    g, d, t = irr
    x = point["A" if g else "a"]
    y = point["B" if d else "b"]
    return x - y if t else x + y


def check_series(point: dict[str, float] | None = None) -> dict[str, float]:
    point = point or {"a": 0.3, "b": 0.2, "A": 0.4, "B": 0.1}
    values = {irr: linear_value(irr, point) for irr in IRREDUCIBLES}
    total_simple = total_alt = 0.0
    log_p = 0.0
    for ((i, j), eps), cells in LABEL_FIBRES.items():
        x = (-1) ** eps * values[i] * values[j]
        assert abs(x) < 1
        total_simple += len(cells) * harmonic_simple(x)
        total_alt += len(cells) * harmonic_alternating(x)
    for (i, j), mult in [((i, j), 4) for i, j in itertools.combinations(IRREDUCIBLES, 2)] + [
        ((i, i), 2) for i in IRREDUCIBLES
    ]:
        x = values[i] * values[j]
        log_p += mult * math.log(1 - x * x)
    # P = prod_{i<j} (1 - x_ij^2)^4 * prod_i (1 - x_ii^2)^2
    assert math.isclose(total_simple, -log_p, abs_tol=1e-9)
    assert math.isclose(total_alt, log_p, abs_tol=1e-9)
    assert math.isclose(total_alt, -total_simple, abs_tol=1e-9)
    return {"total_simple": total_simple, "total_alternating": total_alt}


# ---------------------------------------------------------------------------
# 6. Huffman indexing
# ---------------------------------------------------------------------------

def huffman_lengths(freqs: dict[object, Fraction]) -> dict[object, int]:
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


def canonical_codes(lengths: dict[object, int], order: list[object]) -> dict[object, str]:
    symbols = sorted(lengths, key=lambda s: (lengths[s], order.index(s)))
    codes: dict[object, str] = {}
    code, prev = 0, 0
    for s in symbols:
        length = lengths[s]
        code <<= length - prev
        codes[s] = format(code, f"0{length}b")
        code += 1
        prev = length
    return codes


IDENTITY_ORDER = sorted(LABEL_FIBRES, key=label_name)
FREQS = {l: Fraction(len(LABEL_FIBRES[l]), 256) for l in IDENTITY_ORDER}
HUFFMAN_LENGTHS = huffman_lengths(FREQS)
HUFFMAN_CODES = canonical_codes(HUFFMAN_LENGTHS, IDENTITY_ORDER)  # type: ignore[arg-type]

CELL_WORD: dict[str, str] = {}
for _l in IDENTITY_ORDER:
    _cells = sorted(LABEL_FIBRES[_l])
    _w = int(math.log2(len(_cells)))
    for _i, _c in enumerate(_cells):
        CELL_WORD[_c] = HUFFMAN_CODES[_l] + (format(_i, f"0{_w}b") if _w else "")


def is_prefix_free(words: list[str]) -> bool:
    return not any(x != y and y.startswith(x) for x in words for y in words)


def check_huffman() -> dict[str, float]:
    assert Counter(HUFFMAN_LENGTHS.values()) == Counter({6: 56, 7: 16})
    assert is_prefix_free(list(HUFFMAN_CODES.values()))
    expected = float(sum(FREQS[l] * len(HUFFMAN_CODES[l]) for l in FREQS))
    entropy = -sum(float(p) * math.log2(float(p)) for p in FREQS.values())
    assert math.isclose(expected, 6.125) and math.isclose(entropy, 6.125)
    for l in FREQS:  # code length + log2(multiplicity) = 8 bits per cell
        assert len(HUFFMAN_CODES[l]) + int(math.log2(len(LABEL_FIBRES[l]))) == 8
    assert all(len(w) == 8 for w in CELL_WORD.values())
    assert len(set(CELL_WORD.values())) == 256 and is_prefix_free(list(CELL_WORD.values()))
    return {"expected_length": expected, "entropy": entropy}


# ---------------------------------------------------------------------------
# 7. Hamming relations
# ---------------------------------------------------------------------------
# Hamming(12,8): shortened Hamming(15,11); parity bits at positions 1, 2, 4, 8.

DATA_POSITIONS = [3, 5, 6, 7, 9, 10, 11, 12]


def hamming128_encode(data: tuple[int, ...]) -> tuple[int, ...]:
    w = [0] * 13
    for pos, bit in zip(DATA_POSITIONS, data):
        w[pos] = bit
    for p in (1, 2, 4, 8):
        w[p] = sum(w[i] for i in range(1, 13) if i & p) % 2
    return tuple(w[1:])


def hamming128_decode(word: tuple[int, ...]) -> tuple[int, ...]:
    w = [0] + list(word)
    syndrome = 0
    for i in range(1, 13):
        if w[i]:
            syndrome ^= i
    if syndrome:
        w[syndrome] ^= 1
    return tuple(w[pos] for pos in DATA_POSITIONS)


def hamming_distance(x, y) -> int:
    return sum(p != q for p, q in zip(x, y))


def class_distance(l1: tuple, l2: tuple) -> int:
    return min(hamming_distance(c1, c2) for c1 in LABEL_FIBRES[l1] for c2 in LABEL_FIBRES[l2])


def check_hamming() -> dict[str, object]:
    data = {c: tuple(int(ch) for ch in c) for c in CELLS}
    words = {c: hamming128_encode(d) for c, d in data.items()}
    assert min(hamming_distance(x, y) for x, y in itertools.combinations(words.values(), 2)) == 3
    for c, w in words.items():  # single-error correction
        for pos in range(12):
            bad = list(w)
            bad[pos] ^= 1
            assert hamming128_decode(tuple(bad)) == data[c]

    # Bit-flip graph on the 72 identities.
    edges = set()
    for c in CELLS:
        for pos in range(8):
            f = c[:pos] + ("1" if c[pos] == "0" else "0") + c[pos + 1:]
            if CELL_LABEL[c] != CELL_LABEL[f]:
                edges.add(frozenset((CELL_LABEL[c], CELL_LABEL[f])))
    degree: Counter = Counter()
    adjacency: dict[tuple, set] = defaultdict(set)
    for e in edges:
        x, y = tuple(e)
        degree[x] += 1
        degree[y] += 1
        adjacency[x].add(y)
        adjacency[y].add(x)
    assert len(edges) == 256
    assert Counter(degree.values()) == Counter({8: 56, 4: 16})  # squares have degree 4

    def eccentricity(src: tuple) -> int:
        dist = {src: 0}
        queue = [src]
        for u in queue:
            for w in adjacency[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        return max(dist.values())

    ecc = [eccentricity(l) for l in LABEL_FIBRES]
    assert max(ecc) == 6 and min(ecc) == 4

    # For every pair of squares, a sign choice of p_i p_j (a square root of the
    # product p_i^2 * p_j^2) is a Hamming midpoint of the two squares.
    both = one = 0
    for i, j in itertools.combinations(IRREDUCIBLES, 2):
        si, sj = ((i, i), 0), ((j, j), 0)
        d = class_distance(si, sj)
        mid = [d % 2 == 0 and class_distance(si, (tuple(sorted((i, j))), e)) == d // 2
               and class_distance((tuple(sorted((i, j))), e), sj) == d // 2 for e in (0, 1)]
        assert any(mid)
        both += all(mid)
        one += (not all(mid))
    assert (both, one) == (16, 12)
    return {"edges": len(edges), "pairs_with_both_midpoints": both, "pairs_with_one_midpoint": one}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main() -> None:
    check_partitions()
    check_labels()
    abelian = check_abelian()
    series = check_series()
    huffman = check_huffman()
    hamming = check_hamming()

    print(f"cells: {len(CELLS)}, distinct identities: {len(LABEL_FIBRES)}")
    print("multiplicities:", dict(Counter(len(v) for v in LABEL_FIBRES.values())))
    print("irreducibles:", ", ".join(irr_name(i) for i in IRREDUCIBLES))
    print("multiplicative group: C2 x Z^8; identities lie in degree 2")
    print("additive lattice: rank 10, index", abelian["index"],
          "quotient invariants", abelian["quotient_invariants"])
    print("series totals:", {k: round(v, 12) for k, v in series.items()})
    print("Huffman: 56 identities of length 6, 16 of length 7;",
          "expected length", huffman["expected_length"], "= entropy", huffman["entropy"])
    print("Hamming(12,8): minimum distance 3; bit-flip graph on identities:", hamming)
    print("\nHuffman codes (first 6 identities):")
    for l in IDENTITY_ORDER[:6]:
        print(f"  {label_name(l):<20} {HUFFMAN_CODES[l]:<8} cells={sorted(LABEL_FIBRES[l])}")


if __name__ == "__main__":
    main()
