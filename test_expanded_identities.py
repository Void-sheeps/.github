import math
import itertools
from collections import Counter
import pytest

from expanded_identities import (
    BINOMIAL_BITS,
    CELL_FORM_1,
    CELL_FORM_2,
    CELL_LABEL,
    CELLS,
    FIBRES_1,
    FIBRES_2,
    HUFFMAN_CODES,
    HUFFMAN_LENGTHS,
    IDENTITY_ORDER,
    IRR_INDEX,
    IRREDUCIBLES,
    LABEL_FIBRES,
    MONOMIALS,
    binomial_poly,
    build_cell_forms,
    cell_label,
    check_abelian,
    check_hamming,
    check_huffman,
    check_labels,
    check_partitions,
    check_series,
    class_distance,
    fibres_of,
    g_binomial,
    g_label,
    gmul,
    hamming_distance,
    hamming128_decode,
    hamming128_encode,
    harmonic_alternating,
    harmonic_simple,
    irr_name,
    irr_of,
    is_prefix_free,
    label_name,
    linear_value,
    poly_add,
    poly_key,
    poly_mul,
    poly_scale,
    smith_diagonal,
    variable,
)


def test_poly_arithmetic():
    x = variable(0, 2)
    y = variable(1, 2)
    assert x == {(1, 0): 1}
    assert y == {(0, 1): 1}

    # x + y
    x_plus_y = poly_add(x, y)
    assert x_plus_y == {(1, 0): 1, (0, 1): 1}

    # 3 * x
    three_x = poly_scale(3, x)
    assert three_x == {(1, 0): 3}

    # (x + y) * (x - y) = x^2 - y^2
    neg_y = poly_scale(-1, y)
    x_minus_y = poly_add(x, neg_y)
    prod = poly_mul(x_plus_y, x_minus_y)
    assert prod == {(2, 0): 1, (0, 2): -1}
    assert poly_key(prod) == (((0, 2), -1), ((2, 0), 1))


def test_binomial_bits_and_cells():
    assert len(BINOMIAL_BITS) == 16
    assert len(CELLS) == 256
    assert len(CELL_FORM_1) == 256
    assert len(CELL_FORM_2) == 256


def test_partitions_and_equivalence():
    check_partitions()
    assert len(FIBRES_1) == 72
    assert len(FIBRES_2) == 72
    # Verify fibre multiplicities
    mults = Counter(len(v) for v in FIBRES_1.values())
    assert mults == Counter({4: 56, 2: 16})


def test_irreducibles_and_labels():
    check_labels()
    assert len(IRREDUCIBLES) == 8
    for irr in IRREDUCIBLES:
        assert irr in IRR_INDEX
        name = irr_name(irr)
        assert isinstance(name, str)

    # Check cell label structure
    for cell, label in list(CELL_LABEL.items())[:10]:
        (i, j), eps = label
        assert i in IRREDUCIBLES
        assert j in IRREDUCIBLES
        assert eps in (0, 1)
        l_name = label_name(label)
        assert l_name.startswith("+") or l_name.startswith("-")


def test_abelian_structure_and_smith_diagonal():
    res = check_abelian()
    assert res["index"] == 32
    assert res["quotient_invariants"] == [2, 2, 2, 4]

    # Test smith_diagonal on simple matrices
    diag = smith_diagonal([[2, 0], [0, 4]])
    assert sorted(diag) == [2, 4]


def test_expanded_series():
    series = check_series()
    assert "total_simple" in series
    assert "total_alternating" in series
    assert math.isclose(series["total_simple"], -series["total_alternating"])

    # Custom point
    pt = {"a": 0.1, "b": 0.05, "A": 0.2, "B": 0.08}
    series_pt = check_series(pt)
    assert math.isclose(series_pt["total_simple"], -series_pt["total_alternating"])


def test_huffman():
    huff = check_huffman()
    assert huff["expected_length"] == 6.125
    assert huff["entropy"] == 6.125
    assert is_prefix_free(list(HUFFMAN_CODES.values()))
    lengths = Counter(len(code) for code in HUFFMAN_CODES.values())
    assert lengths == Counter({6: 56, 7: 16})


def test_hamming_relations():
    ham = check_hamming()
    assert ham["edges"] == 256
    assert ham["pairs_with_both_midpoints"] == 16
    assert ham["pairs_with_one_midpoint"] == 12

    # Test hamming128 encode / decode
    data = (1, 0, 1, 1, 0, 0, 1, 0)
    codeword = hamming128_encode(data)
    assert len(codeword) == 12
    decoded = hamming128_decode(codeword)
    assert decoded == data

    # With 1 bit flipped
    for pos in range(12):
        corrupted = list(codeword)
        corrupted[pos] ^= 1
        decoded_corrupted = hamming128_decode(tuple(corrupted))
        assert decoded_corrupted == data
