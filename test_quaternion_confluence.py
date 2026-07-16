import pytest
from quaternion_confluence import (
    inv, free_reduce, make_shortlex, Rule, rewrite_step, normal_form,
    overlaps, critical_pairs, knuth_bendix, discover_group_elements,
    format_word
)

def test_inv():
    assert inv('a') == 'A'
    assert inv('A') == 'a'
    assert inv('b') == 'B'
    assert inv('B') == 'b'


def test_free_reduce():
    assert free_reduce(('a', 'A')) == ()
    assert free_reduce(('a', 'b', 'B', 'A')) == ()
    assert free_reduce(('a', 'b', 'A', 'B')) == ('a', 'b', 'A', 'B')
    assert free_reduce(('a', 'a', 'A', 'b')) == ('a', 'b')


def test_shortlex():
    cmp_fn = make_shortlex(['a', 'b', 'A', 'B'])

    # Same length, same characters
    assert cmp_fn(('a',), ('a',)) == 0

    # Different lengths
    assert cmp_fn(('a',), ('a', 'b')) == -1
    assert cmp_fn(('a', 'b'), ('a',)) == 1

    # Same length, alphabetical rank
    assert cmp_fn(('a',), ('b',)) == -1
    assert cmp_fn(('b',), ('a',)) == 1
    assert cmp_fn(('a', 'b'), ('a', 'A')) == -1


def test_rule_repr_and_str():
    rule = Rule(('a', 'a'), ('b',), "origin_desc")
    assert repr(rule) == "aa -> b"
    assert str(rule) == "aa -> b"

    rule_empty = Rule((), (), "identity")
    assert repr(rule_empty) == "e -> e"


def test_rewrite_step_and_normal_form():
    rules = [
        Rule(('a', 'a'), (), "a^2 = 1"),
        Rule(('b', 'b'), (), "b^2 = 1")
    ]
    # Simple reduction
    res, rule = rewrite_step(('a', 'a', 'b'), rules)
    assert res == ('b',)
    assert rule.lhs == ('a', 'a')

    # Normal form with multiple reductions
    nf, trace = normal_form(('a', 'a', 'b', 'b', 'a'), rules)
    assert nf == ('a',)
    assert len(trace) == 2


def test_overlaps():
    # Suffix-prefix overlap: 'aba' and 'b' -> overlap is 'aba'
    # Inclusion overlap: 'aba' contains 'b'
    # Suffix of 'aba' (a) overlaps prefix of 'aba' (a)
    overlaps_list = list(overlaps(('a', 'b', 'a'), ('a', 'b', 'a')))
    assert len(overlaps_list) > 0


def test_critical_pairs():
    rules = [
        Rule(('a', 'b'), ('c',), "ab = c"),
        Rule(('b', 'd'), ('e',), "bd = e")
    ]
    # 'abd' has left-overlap 'ab' and right-overlap 'bd'
    # s = c d, t = a e
    pairs = critical_pairs(rules)
    has_pair = False
    for s, t, desc in pairs:
        if (s == ('c', 'd') and t == ('a', 'e')) or (s == ('a', 'e') and t == ('c', 'd')):
            has_pair = True
    assert has_pair


def test_knuth_bendix_confluence():
    letter_order = ['a', 'b', 'A', 'B']
    cmp_fn = make_shortlex(letter_order)

    r0 = Rule(('a', 'a', 'a', 'a'), (), "a^4 = 1")
    r1 = Rule(('b', 'b'), ('a', 'a'), "b^2 = a^2")
    r2 = Rule(('a', 'b', 'a'), ('b',), "aba = b")

    inv_rules = [
        Rule(('a', 'A'), (), "a A = 1"),
        Rule(('A', 'a'), (), "A a = 1"),
        Rule(('b', 'B'), (), "b B = 1"),
        Rule(('B', 'b'), (), "B b = 1"),
    ]

    initial_rules = [r0, r1, r2] + inv_rules
    final_rules = knuth_bendix(initial_rules, cmp_fn, verbose=False)

    # After completion, normal_form should be unique for equivalent words
    # e.g., b^2 and a^2 should reduce to the same normal form
    nf_b2, _ = normal_form(('b', 'b'), final_rules)
    nf_a2, _ = normal_form(('a', 'a'), final_rules)
    assert nf_b2 == nf_a2


def test_discover_group_elements():
    letter_order = ['a', 'b', 'A', 'B']
    cmp_fn = make_shortlex(letter_order)

    r0 = Rule(('a', 'a', 'a', 'a'), (), "a^4 = 1")
    r1 = Rule(('b', 'b'), ('a', 'a'), "b^2 = a^2")
    r2 = Rule(('a', 'b', 'a'), ('b',), "aba = b")

    inv_rules = [
        Rule(('a', 'A'), (), "a A = 1"),
        Rule(('A', 'a'), (), "A a = 1"),
        Rule(('b', 'B'), (), "b B = 1"),
        Rule(('B', 'b'), (), "B b = 1"),
    ]

    initial_rules = [r0, r1, r2] + inv_rules
    final_rules = knuth_bendix(initial_rules, cmp_fn, verbose=False)

    elements = discover_group_elements(['a', 'b'], final_rules, cmp_fn)
    # Q_8 has exactly 8 elements
    assert len(elements) == 8

    # Checking some canonical normal forms
    assert () in elements  # Identity
    assert ('a',) in elements
    assert ('b',) in elements


def test_format_word():
    assert format_word(()) == '1'
    assert format_word(('a',)) == 'a'
    assert format_word(('A',)) == 'a^-1'
    assert format_word(('a', 'a')) == 'a^2'
    assert format_word(('A', 'A')) == 'a^-2'
    assert format_word(('a', 'b')) == 'ab'
    assert format_word(('a', 'b', 'B')) == 'abb^-1'  # without free reduction
