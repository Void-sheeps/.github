from __future__ import annotations
import math
import pytest
from meta_system import (
    MetaSystem,
    Node,
    ProvenanceMismatch,
    trachtenberg_single,
    _fn_mul_trachtenberg,
    _fn_mul_digit_convolution,
    _fn_gcd,
    _fn_sin,
    _fn_cos,
    _fn_tan,
    _fn_limit,
    _fn_derivative,
    _fn_integral,
)


def test_meta_system_init_and_acyclic():
    """Verify that a clean MetaSystem initializes and acyclic check succeeds."""
    ms = MetaSystem()
    assert isinstance(ms.nodes, dict)
    assert len(ms.nodes) > 0


def test_cycle_detection():
    """Verify that acyclic check correctly raises ValueError when a cycle exists."""
    ms = MetaSystem()
    # Artificially introduce a dependency cycle
    ms.nodes["add"].depends_on.add("tan")
    ms.nodes["tan"].depends_on.add("add")

    with pytest.raises(ValueError, match="cycle detected"):
        ms._validate_acyclic()


def test_topological_layers():
    """Verify that topological sorted layers have correct dependency relationships."""
    ms = MetaSystem()
    layers = ms.topological_layers()

    # Check that all nodes are present in layers
    all_layered_nodes = [node for layer in layers for node in layer]
    assert len(all_layered_nodes) == len(ms.nodes)
    assert set(all_layered_nodes) == set(ms.nodes.keys())

    # Check that dependencies appear in earlier layers
    resolved = set()
    for layer in layers:
        for node_key in layer:
            node = ms.nodes[node_key]
            assert node.depends_on.issubset(resolved), f"Prerequisites for {node_key} not resolved in earlier layers"
        resolved.update(layer)


def test_trachtenberg_multiplication_logic():
    """Test standard individual Trachtenberg single and multi-digit cases."""
    # Test trachtenberg_single with various digits
    assert trachtenberg_single(4321, 0) == 0
    assert trachtenberg_single(4321, 1) == 4321
    assert trachtenberg_single(4321, 2) == 8642
    assert trachtenberg_single(4321, 5) == 21605
    assert trachtenberg_single(4321, 6) == 25926
    assert trachtenberg_single(4321, 7) == 30247
    assert trachtenberg_single(4321, 9) == 38889

    # Multi-digit trachtenberg
    assert _fn_mul_trachtenberg(4321, 567) == 2450007
    assert _fn_mul_trachtenberg(12345, 9876) == 121919220
    assert _fn_mul_trachtenberg(0, 999) == 0
    assert _fn_mul_trachtenberg(999, 0) == 0


def test_individual_math_algorithms():
    """Test calculus, trigonometry, and arithmetic functions."""
    # GCD
    assert _fn_gcd(48, 18) == 6
    assert _fn_gcd(101, 103) == 1

    # Trig approximations
    assert abs(_fn_sin(math.pi / 6) - 0.5) < 1e-5
    assert abs(_fn_cos(math.pi / 3) - 0.5) < 1e-5
    assert abs(_fn_tan(_fn_sin(0.5), _fn_cos(0.5)) - math.tan(0.5)) < 1e-5

    # Limit
    f = lambda x: (x**2 - 1) / (x - 1) if x != 1 else 0
    assert abs(_fn_limit(f, 1.0) - 2.0) < 1e-5

    # Derivative
    g = lambda x: x**3
    assert abs(_fn_derivative(g, 2.0) - 12.0) < 1e-4

    # Integral
    h = lambda x: x**2
    assert abs(_fn_integral(h, 0.0, 1.0, n=1000) - 1/3) < 1e-4


def test_provenance_verification():
    """Test that provenance verification successfully validates origins and raises mismatch exceptions."""
    ms = MetaSystem()

    # Test valid schoolbook
    report_school = ms.verify_provenance("mul_digit_convolution", 4321, 567)
    assert report_school["claimed_origin"] == "schoolbook"
    assert report_school["verified"] is True
    assert report_school["fingerprint"]["avoided"] == 0

    # Test valid trachtenberg
    report_trach = ms.verify_provenance("mul_trachtenberg", 4321, 567)
    assert report_trach["claimed_origin"] == "trachtenberg"
    assert report_trach["verified"] is True
    assert report_trach["fingerprint"]["avoided"] > 0

    # Mismatch scenario: manually corrupt claimed origin to expect mismatch
    ms.nodes["mul_digit_convolution"].claimed_origin = "trachtenberg"
    with pytest.raises(ProvenanceMismatch, match="Provenance verification failed for node 'mul_digit_convolution'"):
        ms.verify_provenance("mul_digit_convolution", 4321, 567)


def test_system_parallel_run():
    """Test parallel layer execution with resolution of positional and keyword references."""
    ms = MetaSystem()

    calls = {
        "sin": ((0.5,), {}),
        "cos": ((0.5,), {}),
        "tan": (("sin", "cos"), {}),
        "add": ((3, 4), {}),
        "mul": ((3, 4), {}),
        "div": ((10, 4), {}),
        "gcd": ((48, 18), {}),
        "distance": (((0, 0), (3, 4)), {}),
        "derivative": ((lambda x: x ** 2, 3.0), {}),
        "integral": ((lambda x: x ** 2, 0, 1), {}),
        "mul_digit_convolution": ((4321, 567), {}),
        "mul_trachtenberg": ((4321, 567), {}),
    }

    results = ms.run(calls, workers=4)

    assert results["add"] == 7
    assert results["mul"] == 12
    assert results["div"] == 2.5
    assert results["gcd"] == 6
    assert results["mul_digit_convolution"] == 2450007
    assert results["mul_trachtenberg"] == 2450007
    assert abs(results["distance"] - 5.0) < 1e-9
    assert abs(results["tan"] - (results["sin"] / results["cos"])) < 1e-9
    assert abs(results["derivative"] - 6.0) < 1e-5
    assert abs(results["integral"] - 1/3) < 1e-4
