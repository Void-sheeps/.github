import math
import itertools
from collections import Counter
import pytest

from binomial_identities import (
    BINOMIAL_BITS,
    CELL_IDENTITY,
    CELL_WORD,
    CELLS,
    EXPECTED_MATRIX,
    EXPECTED_MULTIPLICITY,
    FIBRES,
    FORM_NAMES,
    FORM_OF,
    G_BINOMIAL,
    G_IDENTITY,
    HAMMING_WORD,
    HUFFMAN_CODES,
    HUFFMAN_LENGTHS,
    IDENTITY_ORDER,
    LATTICE_BASIS,
    MATRIX,
    ORDER,
    WORD_CELL,
    action_on_identities,
    binomial_from_bits,
    bits,
    check_additive_lattice,
    check_bit_homomorphisms,
    check_hamming,
    check_huffman,
    check_matrix,
    check_multiplicative_group,
    check_series,
    check_symmetries,
    class_distance,
    det3,
    ginv,
    gmul,
    gpow,
    hamming_distance,
    hamming74_decode,
    hamming74_encode,
    harmonic_alternating,
    harmonic_simple,
    identity_of_cell,
    is_prefix_free,
    multiply,
    series_over_matrix,
    smith_invariants_3x3,
    substitute,
    summary_table,
    syndrome,
    value,
    word_string,
    xor_cells,
)


def test_binomial_bits_and_order():
    assert len(ORDER) == 4
    assert set(BINOMIAL_BITS.keys()) == set(ORDER)
    assert BINOMIAL_BITS["u"] == (0, 0)
    assert BINOMIAL_BITS["v"] == (0, 1)
    assert BINOMIAL_BITS["-v"] == (1, 0)
    assert BINOMIAL_BITS["-u"] == (1, 1)


def test_matrix_construction_and_properties():
    check_matrix()
    assert len(MATRIX) == 4
    for row in MATRIX:
        assert len(row) == 4
    assert MATRIX == EXPECTED_MATRIX

    # Multiplicity
    counts = {k: len(v) for k, v in FIBRES.items()}
    assert counts == EXPECTED_MULTIPLICITY
    assert sum(counts.values()) == 16


def test_forms_and_multiplication():
    # Test binomial multiplying
    # (a + b) * (a + b) = a^2 + 2ab + b^2 -> (1, 2, 1) -> "A"
    p = binomial_from_bits(0, 0)
    q = binomial_from_bits(0, 0)
    assert p == (1, 1)
    assert q == (1, 1)
    res = multiply(p, q)
    assert res == (1, 2, 1)
    assert FORM_NAMES[res] == "A"

    # (a + b) * (a - b) = a^2 - b^2 -> (1, 0, -1) -> "B"
    res2 = multiply((1, 1), (1, -1))
    assert res2 == (1, 0, -1)
    assert FORM_NAMES[res2] == "B"


def test_series_evaluation():
    res = check_series(0.3, 0.2)
    assert "total_simple" in res
    assert "total_alternating" in res
    assert "minus_ln_P" in res
    assert "ln_P" in res

    # Single term evaluation checks
    val_a = value("A", 0.3, 0.2)
    assert math.isclose(val_a, 0.25)
    assert math.isclose(harmonic_simple(val_a, 500), -math.log(1 - val_a), abs_tol=1e-7)
    assert math.isclose(harmonic_alternating(val_a, 500), math.log(1 + val_a), abs_tol=1e-7)


def test_multiplicative_group():
    check_multiplicative_group()
    # Check group operations
    # gmul, ginv, gpow
    e = (1, 0, 0)
    u = G_BINOMIAL["u"]
    assert gmul(u, ginv(u)) == e
    assert gpow(u, 0) == e
    assert gpow(u, 3) == gmul(u, gmul(u, u))
    assert gpow(u, -2) == gpow(ginv(u), 2)


def test_additive_lattice():
    check_additive_lattice()
    # Check det3 and Smith invariants
    d = det3(LATTICE_BASIS)
    assert abs(d) == 8
    invs = smith_invariants_3x3(LATTICE_BASIS)
    assert invs == (1, 2, 4)


def test_bit_homomorphisms_and_syndrome():
    check_bit_homomorphisms()
    assert syndrome("0000") == (0, 0)
    assert syndrome("0001") == (0, 1)
    assert syndrome("0010") == (1, 0)
    assert syndrome("0011") == (1, 1)

    assert xor_cells("0101", "1010") == "1111"
    assert xor_cells("0000", "1111") == "1111"


def test_d4_symmetries():
    sym = check_symmetries()
    assert sym["effective_group_order"] == 4
    assert len(sym["kernel"]) == 2

    # Verify identity action under identity matrix
    identity_matrix = (1, 0, 0, 1)
    assert action_on_identities(identity_matrix) == tuple(FORM_OF.keys())


def test_summary_table():
    tbl = summary_table()
    assert len(tbl) == 6
    names = [row[0] for row in tbl]
    assert set(names) == {"A", "-A", "E", "-E", "B", "-B"}


def test_huffman_coding():
    res = check_huffman()
    assert math.isclose(res["expected_length"], 2.5)
    assert math.isclose(res["entropy"], 2.5)
    assert is_prefix_free(HUFFMAN_CODES)
    assert is_prefix_free(CELL_WORD)
    assert len(CELL_WORD) == 16
    assert len(WORD_CELL) == 16
    for cell, word in CELL_WORD.items():
        assert WORD_CELL[word] == cell


def test_hamming_relations():
    res = check_hamming()
    assert "bipartite_edges" in res
    assert len(res["bipartite_edges"]) == 8

    # Test single-error detection and correction
    d = (1, 0, 1, 1)
    encoded = hamming74_encode(d)
    assert len(encoded) == 7
    decoded, err_pos = hamming74_decode(encoded)
    assert decoded == d
    assert err_pos == 0

    # Introduce an error at pos 3 (index 2)
    corrupted = list(encoded)
    corrupted[2] ^= 1
    decoded_corr, err_pos_corr = hamming74_decode(tuple(corrupted))
    assert decoded_corr == d
    assert err_pos_corr == 3

    # Test word string
    assert word_string(encoded) == "".join(map(str, encoded))
    assert hamming_distance("0000", "0110") == 2
