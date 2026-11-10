import math
import itertools
from collections import Counter
import pytest

from expanded_identities_substituted import (
    FACTORIZATION,
    MONOMIALS_2,
    ONE,
    PRIMES,
    binomial_poly_2,
    check_additive,
    check_extra_relation,
    check_factorization,
    check_multiplicative,
    check_series,
    from_exponents,
    minus,
    numeric_values,
    plus,
)


def test_primes_definition():
    assert len(PRIMES) == 9
    prime_names = [name for name, _ in PRIMES]
    assert prime_names == ["2", "a+b", "a-b", "a", "b", "A+b", "A-b", "a+B", "a-B"]


def test_factorization_and_check():
    assert len(FACTORIZATION) == 8
    check_factorization()


def test_multiplicative_structure():
    res = check_multiplicative()
    assert res["ambient"] == "C2 x Z^9"
    assert res["generated"] == "C2 x Z^8"
    assert res["identity_index"] == 2


def test_additive_structure():
    res = check_additive()
    assert res["rank"] == 9
    assert res["index"] == 256
    assert res["quotient_invariants"] == [2, 2, 2, 2, 4, 4]


def test_extra_relations():
    check_extra_relation()


def test_series_evaluation():
    res = check_series(0.2, 0.1)
    assert "total_simple" in res
    assert "total_alternating" in res
    assert math.isclose(res["total_simple"], -res["total_alternating"])

    vals = numeric_values(0.2, 0.1)
    assert len(vals) == 8
