"""
AERIS Canonical Flight Parameters and Standard Units Specification.

Authoritative catalog of canonical flight-test telemetry parameters,
physical units, human-readable labels, and standard telemetry aliases.
"""

from __future__ import annotations

from typing import NamedTuple


class CanonicalParameter(NamedTuple):
    """Immutable specification of a canonical flight-test telemetry parameter."""

    parameter_id: str
    display_name: str
    canonical_unit: str | None
    is_timestamp: bool = False
    aliases: tuple[str, ...] = ()


CANONICAL_PARAMETERS: list[CanonicalParameter] = [
    CanonicalParameter(
        parameter_id="timestamp",
        display_name="Timestamp (Primary Time)",
        canonical_unit="s",
        is_timestamp=True,
        aliases=("time", "timestamp", "utc", "utc_time", "datetime", "date_time"),
    ),
    CanonicalParameter(
        parameter_id="indicated_airspeed",
        display_name="Indicated Airspeed (IAS)",
        canonical_unit="knots",
        aliases=(
            "airspeed_kts",
            "ias",
            "ias_kts",
            "indicated_airspeed",
            "cas",
            "cas_kts",
            "airspeed",
            "flight_speed",
            "airspeed_knots",
        ),
    ),
    CanonicalParameter(
        parameter_id="true_airspeed",
        display_name="True Airspeed (TAS)",
        canonical_unit="knots",
        aliases=("tas", "tas_kts", "true_airspeed", "true_airspeed_kts"),
    ),
    CanonicalParameter(
        parameter_id="pressure_altitude",
        display_name="Pressure Altitude",
        canonical_unit="ft",
        aliases=(
            "altitude_ft",
            "palt",
            "press_alt",
            "pressure_altitude",
            "alt_ft",
            "p_alt",
            "baro_altitude",
        ),
    ),
    CanonicalParameter(
        parameter_id="pitch_attitude",
        display_name="Pitch Attitude",
        canonical_unit="deg",
        aliases=("pitch_deg", "pitch", "pitch_attitude", "theta", "pitch_att"),
    ),
    CanonicalParameter(
        parameter_id="roll_attitude",
        display_name="Roll Attitude",
        canonical_unit="deg",
        aliases=("roll_deg", "roll", "roll_attitude", "phi", "roll_att", "bank_deg", "bank_angle"),
    ),
    CanonicalParameter(
        parameter_id="yaw_attitude",
        display_name="Yaw / Heading Attitude",
        canonical_unit="deg",
        aliases=("yaw_deg", "heading_deg", "heading", "yaw", "psi", "true_heading"),
    ),
    CanonicalParameter(
        parameter_id="pitch_rate",
        display_name="Pitch Angular Rate (q)",
        canonical_unit="deg/s",
        aliases=("pitch_rate", "pitch_rate_deg_s", "pitch_rate_dps", "q", "q_deg_s", "q_dps"),
    ),
    CanonicalParameter(
        parameter_id="roll_rate",
        display_name="Roll Angular Rate (p)",
        canonical_unit="deg/s",
        aliases=("roll_rate", "roll_rate_deg_s", "roll_rate_dps", "p", "p_deg_s", "p_dps"),
    ),
    CanonicalParameter(
        parameter_id="yaw_rate",
        display_name="Yaw Angular Rate (r)",
        canonical_unit="deg/s",
        aliases=("yaw_rate", "yaw_rate_deg_s", "yaw_rate_dps", "r", "r_deg_s", "r_dps"),
    ),
    CanonicalParameter(
        parameter_id="angle_of_attack",
        display_name="Angle of Attack (alpha)",
        canonical_unit="deg",
        aliases=("aoa", "aoa_deg", "alpha", "angle_of_attack"),
    ),
    CanonicalParameter(
        parameter_id="sideslip_angle",
        display_name="Sideslip Angle (beta)",
        canonical_unit="deg",
        aliases=("aos", "beta", "sideslip_deg", "sideslip_angle", "beta_deg"),
    ),
    CanonicalParameter(
        parameter_id="vertical_acceleration",
        display_name="Normal Load Factor (Nz)",
        canonical_unit="g",
        aliases=(
            "nz",
            "normal_acceleration",
            "vertical_acceleration",
            "g_load",
            "nz_g",
            "accel_z",
        ),
    ),
    CanonicalParameter(
        parameter_id="dynamic_pressure",
        display_name="Dynamic Pressure (qbar)",
        canonical_unit="psf",
        aliases=("qbar", "q_bar", "dynamic_pressure", "qbar_psf"),
    ),
    CanonicalParameter(
        parameter_id="mach_number",
        display_name="Mach Number (M)",
        canonical_unit="mach",
        aliases=("mach", "mach_number", "m"),
    ),
    CanonicalParameter(
        parameter_id="engine_n1",
        display_name="Engine Fan Speed (N1)",
        canonical_unit="%",
        aliases=("engine_rpm", "rpm", "n1", "engine_n1", "fan_speed"),
    ),
]

CANONICAL_BY_ID: dict[str, CanonicalParameter] = {p.parameter_id: p for p in CANONICAL_PARAMETERS}
CANONICAL_UNITS: dict[str, str | None] = {
    p.parameter_id: p.canonical_unit for p in CANONICAL_PARAMETERS
}
CANONICAL_LABELS: dict[str, str] = {p.parameter_id: p.display_name for p in CANONICAL_PARAMETERS}


def infer_unit_from_column_name(column_name: str) -> str | None:
    """Infer physical engineering unit from column name suffix.

    Returns canonical unit string if known, otherwise None. Never fabricates.
    """
    col = column_name.lower().strip()
    if col.endswith(("_kts", "_knot", "_knots")):
        return "knots"
    if col.endswith(("_ft", "_feet")):
        return "ft"
    if col.endswith(("_deg_s", "_deg/s", "_dps")):
        return "deg/s"
    if col.endswith(("_deg", "_degs", "_degrees")):
        return "deg"
    if col.endswith(("_psf",)):
        return "psf"
    if col.endswith(("_psi",)):
        return "psi"
    if col.endswith(("_rpm",)):
        return "rpm"
    if col.endswith(("_g",)):
        return "g"
    if col.endswith(("_mps", "_m_s", "_m/s")):
        return "m/s"
    if col.endswith(("_m",)):
        return "m"
    if col.endswith(("_pct", "_percent")):
        return "%"
    if col.endswith(("_c", "_degc")):
        return "degC"
    if col.endswith(("_f", "_degf")):
        return "degF"
    return None


def suggest_canonical_parameter(column_name: str) -> tuple[str, str | None, bool]:
    """Suggest canonical parameter ID, canonical unit, and timestamp flag for a raw column.

    Returns
    -------
    tuple[str, str | None, bool]
        (suggested_parameter_id, canonical_unit, is_timestamp)
    """
    col = column_name.lower().strip()

    # 1. Exact canonical ID match
    if col in CANONICAL_BY_ID:
        p = CANONICAL_BY_ID[col]
        return p.parameter_id, p.canonical_unit, p.is_timestamp

    # 2. Match known alias
    for p in CANONICAL_PARAMETERS:
        if col in p.aliases:
            return p.parameter_id, p.canonical_unit, p.is_timestamp

    # 3. Timestamp heuristic
    if "time" in col or "stamp" in col:
        return "timestamp", "s", True

    # 4. Identity fallback with unit suffix detection
    inferred_unit = infer_unit_from_column_name(col)
    return col.replace(" ", "_"), inferred_unit, False
