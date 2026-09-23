import re
from typing import Any


COULOMB_CONSTANT = 8.99e9
MAX_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
CHARGE_PATTERN = re.compile(
    rf"(?P<value>{MAX_NUMBER})\s*(?P<unit>μC|µC|uC|mC|nC|C)\b",
    re.IGNORECASE,
)
DISTANCE_PATTERN = re.compile(
    rf"(?P<value>{MAX_NUMBER})\s*(?P<unit>cm|km|m)\b",
    re.IGNORECASE,
)


def _failure(reason: str) -> dict[str, Any]:
    return {"success": False, "reason": reason}


def _convert_charge(value: float, unit: str) -> float:
    factors = {"c": 1.0, "mc": 1e-3, "μc": 1e-6, "µc": 1e-6, "uc": 1e-6, "nc": 1e-9}
    return value * factors[unit.lower()]


def _convert_distance(value: float, unit: str) -> float:
    factors = {"m": 1.0, "cm": 1e-2, "km": 1e3}
    return value * factors[unit.lower()]


def calculate_electric_field(question: str) -> dict[str, Any]:
    """Calculate E = k|q|/r² for a point charge when both values are explicit."""
    charges = CHARGE_PATTERN.findall(question)
    distances = DISTANCE_PATTERN.findall(question)
    if not charges:
        return _failure("A charge value with a supported unit is required.")
    if not distances:
        return _failure("A distance value with a supported unit is required.")
    if len(charges) > 1 or len(distances) > 1:
        return _failure("The question contains multiple charge or distance values.")

    try:
        charge_value, charge_unit = charges[0]
        distance_value, distance_unit = distances[0]
        charge = _convert_charge(float(charge_value), charge_unit)
        distance = _convert_distance(float(distance_value), distance_unit)
    except (KeyError, ValueError):
        return _failure("Charge or distance contains an invalid numeric value.")
    if distance == 0:
        return _failure("Distance must be non-zero.")

    result = COULOMB_CONSTANT * abs(charge) / distance**2
    return {
        "success": True,
        "formula": "E = k|q|/r²",
        "known_values": {"q": charge, "r": distance, "k": COULOMB_CONSTANT},
        "substitution": f"E = ({COULOMB_CONSTANT:g} × {abs(charge):g}) / ({distance:g})²",
        "result": result,
        "unit": "N/C",
    }
