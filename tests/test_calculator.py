from src.study_assistant.calculator import calculate_electric_field
import pytest


def test_electric_field_calculation():
    result = calculate_electric_field(
        "A point charge of 2 μC is 3 m away. Calculate the electric field."
    )
    assert result["success"] is True
    assert result["result"] == pytest.approx(1997.7777777777778)


def test_charge_unit_conversion():
    result = calculate_electric_field("q = 2 μC and r = 3 m")
    assert result["known_values"]["q"] == 2e-6


def test_structured_calculation_output():
    result = calculate_electric_field("q = 2 μC and r = 3 m")
    assert set(("formula", "known_values", "substitution", "result", "unit")) <= result.keys()


def test_missing_distance_returns_controlled_failure():
    result = calculate_electric_field("A point charge of 2 μC is present.")
    assert result["success"] is False


def test_invalid_input_does_not_crash():
    result = calculate_electric_field("Calculate electric field for q = nope")
    assert result["success"] is False


def test_zero_distance_fails_safely():
    result = calculate_electric_field("q = 2 μC and r = 0 m")
    assert result["success"] is False
