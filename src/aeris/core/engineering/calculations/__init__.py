from aeris.core.engineering.calculations.airspeed_rate import calculate_airspeed_rate
from aeris.core.engineering.calculations.altitude_rate import calculate_altitude_rate
from aeris.core.engineering.calculations.dynamic_pressure import calculate_dynamic_pressure
from aeris.core.engineering.calculations.pitch_rate import calculate_pitch_rate
from aeris.core.engineering.calculations.roll_rate import calculate_roll_rate
from aeris.core.engineering.registry import get_calculation_registry


def register_default_calculations() -> None:
    registry = get_calculation_registry()
    registry.register("altitude_rate", calculate_altitude_rate)
    registry.register("airspeed_rate", calculate_airspeed_rate)
    registry.register("pitch_rate", calculate_pitch_rate)
    registry.register("roll_rate", calculate_roll_rate)
    registry.register("dynamic_pressure", calculate_dynamic_pressure)
