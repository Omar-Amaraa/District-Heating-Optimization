"""Unit tests for the MILP and the gas-share computation (run: python tests.py)."""
from milp import optimize_for_given_source, number_of_nodes, cMax, ENR_sourceMaxCap
from pso import calculate_pgaz_posteriori


# ── Helpers ──────────────────────────────────────────────────────────────────

def assert_equal(label, actual, expected):
    if actual != expected:
        print(f"  ✗ {label}: expected {expected}, got {actual}")
    else:
        print(f"  ✓ {label}")

def assert_true(label, condition):
    if condition:
        print(f"  ✓ {label}")
    else:
        print(f"  ✗ {label}: condition is false")

def assert_is_number(label, val):
    if isinstance(val, (int, float)) and val is not None:
        print(f"  ✓ {label}: {val}")
    else:
        print(f"  ✗ {label}: invalid value → {val}")


# ── Tests ────────────────────────────────────────────────────────────────────

def test_optimize_status_optimal():
    """The MILP must find an optimal solution with reasonable parameters."""
    print("\n[test_optimize_status_optimal]")
    result = optimize_for_given_source(source_id=0, P_installed=5000, gaz_part_i=0.3)
    assert_equal("status", result["status"], "Optimal")


def test_optimize_complete_output():
    """Every field of the returned dict must be present and be a number."""
    print("\n[test_optimize_complete_output]")
    result = optimize_for_given_source(source_id=0, P_installed=5000, gaz_part_i=0.3)
    assert_is_number("total_cost",      result["total_cost"])
    assert_is_number("Q_deliverd",      result["Q_deliverd"])
    assert_is_number("Power_renewable", result["Power_renewable"])
    assert_is_number("Power_gaz",       result["Power_gaz"])


def test_optimize_power_renewable_bound():
    """Power_renewable must not exceed ENR_sourceMaxCap."""
    print("\n[test_optimize_power_renewable_bound]")
    result = optimize_for_given_source(source_id=0, P_installed=5000, gaz_part_i=0.3)
    assert_true(
        f"Power_renewable ({result['Power_renewable']}) <= ENR_sourceMaxCap ({ENR_sourceMaxCap})",
        result["Power_renewable"] <= ENR_sourceMaxCap
    )


def test_optimize_q_delivered_positive():
    """Q_deliverd must be positive (heat is actually delivered)."""
    print("\n[test_optimize_q_delivered_positive]")
    result = optimize_for_given_source(source_id=0, P_installed=5000, gaz_part_i=0.3)
    assert_true(
        f"Q_deliverd ({result['Q_deliverd']:.2f}) > 0",
        result["Q_deliverd"] > 0
    )


def test_built_binary():
    """Every value of built must be 0 or 1."""
    print("\n[test_built_binary]")
    result = optimize_for_given_source(source_id=0, P_installed=5000, gaz_part_i=0.3)
    ok = all(
        result["built"][i][j] in (0, 1)
        for i in range(number_of_nodes)
        for j in range(number_of_nodes)
    )
    assert_true("every value of built is 0 or 1", ok)


def test_pgaz_all_renewable():
    """If P_installed <= ENR_sourceMaxCap, the gas share must be 0."""
    print("\n[test_pgaz_all_renewable]")
    part = calculate_pgaz_posteriori(P_installed=1000, source_id=0)
    assert_equal("pgaz with P <= ENR_sourceMaxCap", part, 0)


def test_pgaz_bounds():
    """The gas share must always be between 0 and 1."""
    print("\n[test_pgaz_bounds]")
    for p in [0, 1000, 2000, 3000, 5000, 10000]:
        part = calculate_pgaz_posteriori(P_installed=p, source_id=0)
        assert_true(f"pgaz({p}) in [0,1] → {part:.3f}", 0.0 <= part <= 1.0)


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("UNIT TESTS")
    print("=" * 50)

    test_optimize_status_optimal()
    test_optimize_complete_output()
    test_optimize_power_renewable_bound()
    test_optimize_q_delivered_positive()
    test_built_binary()
    test_pgaz_all_renewable()
    test_pgaz_bounds()

    print("\n" + "=" * 50)
    print("Tests finished.")
    print("=" * 50)
