"""
Comprehensive tests for Engineering Limits domain models, atomic YAML loaders,
unit compatibility, ambiguity/monotonicity checks, and epistemic boundaries.

DISCLAIMER: All test vehicles, parameters, and limits in this module are synthetic and
non-operational. They do NOT establish flight safety, airworthiness, or manufacturer specs.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest
from pydantic import ValidationError

from aeris.core.limits import (
    ConflictingLimitError,
    DuplicateLimitError,
    EngineeringLimitProfile,
    IncompatibleLimitUnitError,
    InvalidLimitProfileError,
    LimitProfileMismatchError,
    LimitSeverity,
    LimitType,
    ParameterLimitDefinition,
    UnknownParameterLimitError,
    load_engineering_limit_profile_from_yaml,
    load_limit_profile_from_yaml,
)
from aeris.core.registry import (
    ParameterDataType,
    ParameterDefinition,
    ParameterRegistry,
)
from aeris.core.vehicle import (
    ConfigurationEnvelope,
    VehicleProfile,
)


@pytest.fixture
def limits_fixtures_dir(fixtures_dir: pathlib.Path) -> pathlib.Path:
    """Return the limits fixtures directory."""
    return fixtures_dir / "limits"


@pytest.fixture
def sample_vehicle_profile() -> VehicleProfile:
    """Return a standard synthetic VehicleProfile for testing."""
    return VehicleProfile(
        profile_id="PROF-AERO-01",
        vehicle_id="AC-AERO-X1",
        profile_version="1.0.0",
        name="Synthetic Aero Testbed Profile",
        supported_parameter_ids=("airspeed", "altitude", "pitch_rate"),
    )


@pytest.fixture
def sample_registry() -> ParameterRegistry:
    """Return a sample ParameterRegistry populated with standard synthetic test parameters."""
    registry = ParameterRegistry()
    registry.register(
        ParameterDefinition(
            parameter_id="airspeed",
            display_name="Airspeed",
            description="Calibrated airspeed",
            physical_quantity="velocity",
            data_type=ParameterDataType.FLOAT,
            canonical_unit="kts",
        )
    )
    registry.register(
        ParameterDefinition(
            parameter_id="altitude",
            display_name="Altitude",
            description="Pressure altitude",
            physical_quantity="distance",
            data_type=ParameterDataType.FLOAT,
            canonical_unit="ft",
        )
    )
    registry.register(
        ParameterDefinition(
            parameter_id="pitch_rate",
            display_name="Pitch Rate",
            description="Body axis pitch angular rate",
            physical_quantity="angular_velocity",
            data_type=ParameterDataType.FLOAT,
            canonical_unit="deg/s",
        )
    )
    registry.register(
        ParameterDefinition(
            parameter_id="unitless_ratio",
            display_name="Unitless Ratio",
            description="Dimensionless test factor",
            physical_quantity="ratio",
            data_type=ParameterDataType.FLOAT,
            canonical_unit=None,  # No canonical unit specified
        )
    )
    return registry


# ---------------------------------------------------------------------------
# 1. Valid Limit Definitions and Profiles
# ---------------------------------------------------------------------------


def test_valid_minimal_upper_bound_limit() -> None:
    limit = ParameterLimitDefinition(
        limit_id="LIM-SPD-MAX",
        parameter_id="airspeed",
        limit_type=LimitType.UPPER_BOUND,
        upper_bound=250.0,
        unit="kts",
        vehicle_id="AC-AERO-X1",
        profile_id="PROF-AERO-01",
        profile_version="1.0.0",
        source_reference="AFM-SEC-2",
    )
    assert limit.limit_id == "LIM-SPD-MAX"
    assert limit.parameter_id == "airspeed"
    assert limit.limit_type == LimitType.UPPER_BOUND
    assert limit.severity == LimitSeverity.EXCEEDANCE
    assert limit.upper_bound == 250.0
    assert limit.lower_bound is None
    assert limit.unit == "kts"
    assert limit.configuration_id is None
    assert limit.is_exceeded(250.1) is True
    assert limit.is_exceeded(250.0) is False
    assert limit.is_exceeded(200.0) is False
    assert limit.calculate_margin(240.0) == 10.0
    assert limit.calculate_margin(260.0) == -10.0


def test_valid_lower_bound_limit_and_margin() -> None:
    limit = ParameterLimitDefinition(
        limit_id="LIM-ALT-MIN",
        parameter_id="altitude",
        limit_type=LimitType.LOWER_BOUND,
        severity=LimitSeverity.WARNING,
        lower_bound=1000.0,
        unit="ft",
        vehicle_id="AC-AERO-X1",
        profile_id="PROF-AERO-01",
        profile_version="1.0.0",
        source_reference="SOP-01",
        description="Minimum terrain clearance buffer",
    )
    assert limit.lower_bound == 1000.0
    assert limit.upper_bound is None
    assert limit.is_exceeded(999.0) is True
    assert limit.is_exceeded(1000.0) is False
    assert limit.is_exceeded(1500.0) is False
    assert limit.calculate_margin(1100.0) == 100.0
    assert limit.calculate_margin(950.0) == -50.0


def test_valid_band_limit_and_margin() -> None:
    limit = ParameterLimitDefinition(
        limit_id="LIM-ALT-BAND",
        parameter_id="altitude",
        limit_type=LimitType.BAND,
        severity=LimitSeverity.CAUTION,
        lower_bound=10000.0,
        upper_bound=20000.0,
        unit="ft",
        vehicle_id="AC-AERO-X1",
        profile_id="PROF-AERO-01",
        profile_version="1.0.0",
        source_reference="CARD-05",
    )
    assert limit.lower_bound == 10000.0
    assert limit.upper_bound == 20000.0
    assert limit.is_exceeded(9999.0) is True
    assert limit.is_exceeded(20001.0) is True
    assert limit.is_exceeded(15000.0) is False
    # Margin is distance to nearest boundary
    assert limit.calculate_margin(12000.0) == 2000.0
    assert limit.calculate_margin(18000.0) == 2000.0
    assert limit.calculate_margin(9500.0) == -500.0


def test_scalar_limit_value_mapping() -> None:
    # Scalar limit_value mapped to upper_bound for UPPER_BOUND
    l_upper = ParameterLimitDefinition(
        limit_id="LIM-1",
        parameter_id="airspeed",
        limit_type=LimitType.UPPER_BOUND,
        limit_value=250.0,
        unit="kts",
        vehicle_id="AC-1",
        profile_id="PROF-1",
        profile_version="1.0",
        source_reference="REF",
    )
    assert l_upper.upper_bound == 250.0
    assert l_upper.lower_bound is None

    # Scalar limit_value mapped to lower_bound for LOWER_BOUND
    l_lower = ParameterLimitDefinition(
        limit_id="LIM-2",
        parameter_id="airspeed",
        limit_type=LimitType.LOWER_BOUND,
        limit_value=60.0,
        unit="kts",
        vehicle_id="AC-1",
        profile_id="PROF-1",
        profile_version="1.0",
        source_reference="REF",
    )
    assert l_lower.lower_bound == 60.0
    assert l_lower.upper_bound is None


# ---------------------------------------------------------------------------
# 2. Invalid Identifiers and Unsupported Limit Types
# ---------------------------------------------------------------------------


def test_invalid_identifiers() -> None:
    # Invalid limit_id
    with pytest.raises(ValidationError, match="alphanumeric"):
        ParameterLimitDefinition(
            limit_id="LIM 1",
            parameter_id="airspeed",
            limit_type=LimitType.UPPER_BOUND,
            upper_bound=200.0,
            unit="kts",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            source_reference="REF",
        )

    # Empty limit_id
    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        ParameterLimitDefinition(
            limit_id="  ",
            parameter_id="airspeed",
            limit_type=LimitType.UPPER_BOUND,
            upper_bound=200.0,
            unit="kts",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            source_reference="REF",
        )


def test_unsupported_limit_type_and_severity() -> None:
    with pytest.raises(ValidationError):
        ParameterLimitDefinition(
            limit_id="LIM-1",
            parameter_id="airspeed",
            limit_type="APPROXIMATE",  # type: ignore[arg-type]
            upper_bound=200.0,
            unit="kts",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            source_reference="REF",
        )

    with pytest.raises(ValidationError):
        ParameterLimitDefinition(
            limit_id="LIM-1",
            parameter_id="airspeed",
            limit_type=LimitType.UPPER_BOUND,
            severity="CRITICAL",  # type: ignore[arg-type]
            upper_bound=200.0,
            unit="kts",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            source_reference="REF",
        )


# ---------------------------------------------------------------------------
# 3. Missing Units and Missing Source References
# ---------------------------------------------------------------------------


def test_missing_or_empty_unit() -> None:
    # Missing unit
    with pytest.raises(ValidationError):
        ParameterLimitDefinition(  # type: ignore[call-arg]
            limit_id="LIM-1",
            parameter_id="airspeed",
            limit_type=LimitType.UPPER_BOUND,
            upper_bound=200.0,
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            source_reference="REF",
        )

    # Empty unit
    with pytest.raises(ValidationError, match="unit cannot be empty"):
        ParameterLimitDefinition(
            limit_id="LIM-1",
            parameter_id="airspeed",
            limit_type=LimitType.UPPER_BOUND,
            upper_bound=200.0,
            unit="   ",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            source_reference="REF",
        )


def test_missing_or_empty_source_reference() -> None:
    # Missing source_reference
    with pytest.raises(ValidationError):
        ParameterLimitDefinition(  # type: ignore[call-arg]
            limit_id="LIM-1",
            parameter_id="airspeed",
            limit_type=LimitType.UPPER_BOUND,
            upper_bound=200.0,
            unit="kts",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
        )

    # Empty source_reference
    with pytest.raises(ValidationError, match="source_reference cannot be empty"):
        ParameterLimitDefinition(
            limit_id="LIM-1",
            parameter_id="airspeed",
            limit_type=LimitType.UPPER_BOUND,
            upper_bound=200.0,
            unit="kts",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            source_reference="  ",
        )


# ---------------------------------------------------------------------------
# 4. Invalid Numeric Values and Contradictory Boundaries
# ---------------------------------------------------------------------------


def test_non_finite_numeric_values() -> None:
    with pytest.raises(ValidationError, match="finite number"):
        ParameterLimitDefinition(
            limit_id="LIM-1",
            parameter_id="airspeed",
            limit_type=LimitType.UPPER_BOUND,
            upper_bound=float("nan"),
            unit="kts",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            source_reference="REF",
        )

    with pytest.raises(ValidationError, match="finite number"):
        ParameterLimitDefinition(
            limit_id="LIM-1",
            parameter_id="airspeed",
            limit_type=LimitType.UPPER_BOUND,
            upper_bound=float("inf"),
            unit="kts",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            source_reference="REF",
        )


def test_contradictory_band_boundaries() -> None:
    # lower >= upper
    with pytest.raises(ValidationError, match="strictly less than upper_bound"):
        ParameterLimitDefinition(
            limit_id="LIM-1",
            parameter_id="altitude",
            limit_type=LimitType.BAND,
            lower_bound=10000.0,
            upper_bound=5000.0,
            unit="ft",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            source_reference="REF",
        )

    with pytest.raises(ValidationError, match="strictly less than upper_bound"):
        ParameterLimitDefinition(
            limit_id="LIM-1",
            parameter_id="altitude",
            limit_type=LimitType.BAND,
            lower_bound=10000.0,
            upper_bound=10000.0,
            unit="ft",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            source_reference="REF",
        )


def test_boundary_type_mismatches() -> None:
    # UPPER_BOUND missing upper_bound
    with pytest.raises(ValidationError, match="UPPER_BOUND limit requires upper_bound"):
        ParameterLimitDefinition(
            limit_id="LIM-1",
            parameter_id="airspeed",
            limit_type=LimitType.UPPER_BOUND,
            unit="kts",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            source_reference="REF",
        )

    # UPPER_BOUND with forbidden lower_bound
    with pytest.raises(ValidationError, match="UPPER_BOUND limit must not specify lower_bound"):
        ParameterLimitDefinition(
            limit_id="LIM-1",
            parameter_id="airspeed",
            limit_type=LimitType.UPPER_BOUND,
            upper_bound=200.0,
            lower_bound=50.0,
            unit="kts",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            source_reference="REF",
        )

    # LOWER_BOUND with forbidden upper_bound
    with pytest.raises(ValidationError, match="LOWER_BOUND limit must not specify upper_bound"):
        ParameterLimitDefinition(
            limit_id="LIM-1",
            parameter_id="airspeed",
            limit_type=LimitType.LOWER_BOUND,
            lower_bound=50.0,
            upper_bound=200.0,
            unit="kts",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            source_reference="REF",
        )


# ---------------------------------------------------------------------------
# 5. Duplicate Limit IDs
# ---------------------------------------------------------------------------


def test_duplicate_limit_ids_rejected() -> None:
    l1 = ParameterLimitDefinition(
        limit_id="LIM-01",
        parameter_id="airspeed",
        limit_type=LimitType.UPPER_BOUND,
        upper_bound=250.0,
        unit="kts",
        vehicle_id="AC-1",
        profile_id="PROF-1",
        profile_version="1.0",
        source_reference="REF",
    )
    l2 = ParameterLimitDefinition(
        limit_id="LIM-01",  # Same limit_id
        parameter_id="altitude",
        limit_type=LimitType.LOWER_BOUND,
        lower_bound=1000.0,
        unit="ft",
        vehicle_id="AC-1",
        profile_id="PROF-1",
        profile_version="1.0",
        source_reference="REF",
    )

    with pytest.raises(DuplicateLimitError, match="Duplicate limit_id found in profile"):
        EngineeringLimitProfile(
            limit_profile_id="LIM-PROF-1",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            limits=(l1, l2),
        )


# ---------------------------------------------------------------------------
# 6. Multiple Limits for the Same Parameter
# ---------------------------------------------------------------------------


def test_valid_multiple_limits_for_same_parameter() -> None:
    # 1. Lower bound (minimum airspeed) + Upper bound (maximum airspeed)
    # 2. Caution upper bound (220 kts) + Warning upper bound (250 kts)
    # 3. Flaps configuration specific limit (140 kts)
    l_min = ParameterLimitDefinition(
        limit_id="LIM-SPD-MIN",
        parameter_id="airspeed",
        limit_type=LimitType.LOWER_BOUND,
        severity=LimitSeverity.WARNING,
        lower_bound=65.0,
        unit="kts",
        vehicle_id="AC-1",
        profile_id="PROF-1",
        profile_version="1.0",
        source_reference="AFM",
    )
    l_caut = ParameterLimitDefinition(
        limit_id="LIM-SPD-CAUT",
        parameter_id="airspeed",
        limit_type=LimitType.UPPER_BOUND,
        severity=LimitSeverity.CAUTION,
        upper_bound=220.0,
        unit="kts",
        vehicle_id="AC-1",
        profile_id="PROF-1",
        profile_version="1.0",
        source_reference="AFM",
    )
    l_warn = ParameterLimitDefinition(
        limit_id="LIM-SPD-WARN",
        parameter_id="airspeed",
        limit_type=LimitType.UPPER_BOUND,
        severity=LimitSeverity.WARNING,
        upper_bound=250.0,
        unit="kts",
        vehicle_id="AC-1",
        profile_id="PROF-1",
        profile_version="1.0",
        source_reference="AFM",
    )
    l_flaps = ParameterLimitDefinition(
        limit_id="LIM-SPD-FLAPS",
        parameter_id="airspeed",
        limit_type=LimitType.UPPER_BOUND,
        severity=LimitSeverity.WARNING,
        upper_bound=140.0,
        unit="kts",
        vehicle_id="AC-1",
        profile_id="PROF-1",
        profile_version="1.0",
        configuration_id="CONF-FLAPS",
        source_reference="AFM",
    )

    profile = EngineeringLimitProfile(
        limit_profile_id="LIM-PROF-1",
        vehicle_id="AC-1",
        profile_id="PROF-1",
        profile_version="1.0",
        limits=(l_min, l_caut, l_warn, l_flaps),
    )

    # General queries across all configurations
    all_spd = profile.get_limits_for_parameter("airspeed", include_all_configurations=True)
    assert len(all_spd) == 4

    # Queries filtered by clean/default configuration
    clean_spd = profile.get_limits_for_parameter("airspeed", configuration_id="CONF-CLEAN")
    # Returns baseline limits (configuration_id is None)
    assert len(clean_spd) == 3

    # Queries filtered by CONF-FLAPS configuration
    flaps_spd = profile.get_limits_for_parameter("airspeed", configuration_id="CONF-FLAPS")
    assert len(flaps_spd) == 4


# ---------------------------------------------------------------------------
# 7. Ambiguous or Overlapping Applicability
# ---------------------------------------------------------------------------


def test_duplicate_scope_definition_rejected() -> None:
    # Two identical scope definitions for the same parameter, configuration, type, and severity
    l1 = ParameterLimitDefinition(
        limit_id="LIM-SPD-1",
        parameter_id="airspeed",
        limit_type=LimitType.UPPER_BOUND,
        severity=LimitSeverity.WARNING,
        upper_bound=250.0,
        unit="kts",
        vehicle_id="AC-1",
        profile_id="PROF-1",
        profile_version="1.0",
        source_reference="AFM",
    )
    l2 = ParameterLimitDefinition(
        limit_id="LIM-SPD-2",
        parameter_id="airspeed",
        limit_type=LimitType.UPPER_BOUND,
        severity=LimitSeverity.WARNING,
        upper_bound=240.0,  # Different value but identical scope key!
        unit="kts",
        vehicle_id="AC-1",
        profile_id="PROF-1",
        profile_version="1.0",
        source_reference="AFM",
    )

    with pytest.raises(ConflictingLimitError, match="Duplicate limit definition for parameter"):
        EngineeringLimitProfile(
            limit_profile_id="LIM-PROF-1",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            limits=(l1, l2),
        )


def test_multiple_bands_in_same_scope_rejected() -> None:
    # Multiple BAND limits in the same parameter/configuration scope (even across severities)
    b1 = ParameterLimitDefinition(
        limit_id="LIM-ALT-B1",
        parameter_id="altitude",
        limit_type=LimitType.BAND,
        severity=LimitSeverity.CAUTION,
        lower_bound=1000.0,
        upper_bound=10000.0,
        unit="ft",
        vehicle_id="AC-1",
        profile_id="PROF-1",
        profile_version="1.0",
        source_reference="AFM",
    )
    b2 = ParameterLimitDefinition(
        limit_id="LIM-ALT-B2",
        parameter_id="altitude",
        limit_type=LimitType.BAND,
        severity=LimitSeverity.WARNING,
        lower_bound=5000.0,
        upper_bound=15000.0,
        unit="ft",
        vehicle_id="AC-1",
        profile_id="PROF-1",
        profile_version="1.0",
        source_reference="AFM",
    )

    with pytest.raises(ConflictingLimitError, match="Multiple overlapping BAND limits"):
        EngineeringLimitProfile(
            limit_profile_id="LIM-PROF-1",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            limits=(b1, b2),
        )


def test_contradictory_severity_thresholds_rejected() -> None:
    # Caution upper bound > Warning upper bound
    caut = ParameterLimitDefinition(
        limit_id="LIM-SPD-C",
        parameter_id="airspeed",
        limit_type=LimitType.UPPER_BOUND,
        severity=LimitSeverity.CAUTION,
        upper_bound=260.0,  # CAUTION higher than WARNING
        unit="kts",
        vehicle_id="AC-1",
        profile_id="PROF-1",
        profile_version="1.0",
        source_reference="AFM",
    )
    warn = ParameterLimitDefinition(
        limit_id="LIM-SPD-W",
        parameter_id="airspeed",
        limit_type=LimitType.UPPER_BOUND,
        severity=LimitSeverity.WARNING,
        upper_bound=250.0,
        unit="kts",
        vehicle_id="AC-1",
        profile_id="PROF-1",
        profile_version="1.0",
        source_reference="AFM",
    )

    with pytest.raises(ConflictingLimitError, match=r"CAUTION upper bound.*cannot exceed WARNING"):
        EngineeringLimitProfile(
            limit_profile_id="LIM-PROF-1",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            limits=(caut, warn),
        )

    # Caution lower bound < Warning lower bound
    caut_low = ParameterLimitDefinition(
        limit_id="LIM-ALT-C",
        parameter_id="altitude",
        limit_type=LimitType.LOWER_BOUND,
        severity=LimitSeverity.CAUTION,
        lower_bound=800.0,  # CAUTION lower than WARNING
        unit="ft",
        vehicle_id="AC-1",
        profile_id="PROF-1",
        profile_version="1.0",
        source_reference="AFM",
    )
    warn_low = ParameterLimitDefinition(
        limit_id="LIM-ALT-W",
        parameter_id="altitude",
        limit_type=LimitType.LOWER_BOUND,
        severity=LimitSeverity.WARNING,
        lower_bound=1000.0,
        unit="ft",
        vehicle_id="AC-1",
        profile_id="PROF-1",
        profile_version="1.0",
        source_reference="AFM",
    )

    with pytest.raises(
        ConflictingLimitError, match=r"CAUTION lower bound.*cannot be below WARNING"
    ):
        EngineeringLimitProfile(
            limit_profile_id="LIM-PROF-1",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            limits=(caut_low, warn_low),
        )


# ---------------------------------------------------------------------------
# 8. Unknown Parameter IDs when Registry Validation is Enabled
# ---------------------------------------------------------------------------


def test_unknown_parameter_id_rejected(sample_registry: ParameterRegistry) -> None:
    l_unknown = ParameterLimitDefinition(
        limit_id="LIM-UNK",
        parameter_id="nonexistent_parameter_xyz",
        limit_type=LimitType.UPPER_BOUND,
        upper_bound=100.0,
        unit="kts",
        vehicle_id="AC-AERO-X1",
        profile_id="PROF-AERO-01",
        profile_version="1.0.0",
        source_reference="DOC",
    )
    profile = EngineeringLimitProfile(
        limit_profile_id="LIM-PROF-1",
        vehicle_id="AC-AERO-X1",
        profile_id="PROF-AERO-01",
        profile_version="1.0.0",
        limits=(l_unknown,),
    )

    with pytest.raises(UnknownParameterLimitError, match="references unknown parameter"):
        profile.validate_registry_compatibility(sample_registry)


# ---------------------------------------------------------------------------
# 9. Incompatible Units where Reliable Validation is Supported
# ---------------------------------------------------------------------------


def test_incompatible_unit_rejected(sample_registry: ParameterRegistry) -> None:
    # airspeed has canonical_unit="kts" in registry; limit declares "m/s"
    l_incompat = ParameterLimitDefinition(
        limit_id="LIM-SPD-MS",
        parameter_id="airspeed",
        limit_type=LimitType.UPPER_BOUND,
        upper_bound=100.0,
        unit="m/s",  # Incompatible unit
        vehicle_id="AC-AERO-X1",
        profile_id="PROF-AERO-01",
        profile_version="1.0.0",
        source_reference="DOC",
    )
    profile = EngineeringLimitProfile(
        limit_profile_id="LIM-PROF-1",
        vehicle_id="AC-AERO-X1",
        profile_id="PROF-AERO-01",
        profile_version="1.0.0",
        limits=(l_incompat,),
    )

    with pytest.raises(
        IncompatibleLimitUnitError, match=r"declares unit 'm/s'.*canonical_unit 'kts'"
    ):
        profile.validate_registry_compatibility(sample_registry)


def test_parameter_without_canonical_unit_permits_explicit_unit(
    sample_registry: ParameterRegistry,
) -> None:
    # unitless_ratio has canonical_unit=None in registry
    l_ratio = ParameterLimitDefinition(
        limit_id="LIM-RATIO",
        parameter_id="unitless_ratio",
        limit_type=LimitType.UPPER_BOUND,
        upper_bound=1.2,
        unit="ratio",
        vehicle_id="AC-AERO-X1",
        profile_id="PROF-AERO-01",
        profile_version="1.0.0",
        source_reference="DOC",
    )
    profile = EngineeringLimitProfile(
        limit_profile_id="LIM-PROF-1",
        vehicle_id="AC-AERO-X1",
        profile_id="PROF-AERO-01",
        profile_version="1.0.0",
        limits=(l_ratio,),
    )
    profile.validate_registry_compatibility(sample_registry)  # Does not raise


# ---------------------------------------------------------------------------
# 10. Vehicle / Profile / Version Mismatches
# ---------------------------------------------------------------------------


def test_internal_limit_vehicle_mismatch_rejected() -> None:
    l_mismatch = ParameterLimitDefinition(
        limit_id="LIM-1",
        parameter_id="airspeed",
        limit_type=LimitType.UPPER_BOUND,
        upper_bound=250.0,
        unit="kts",
        vehicle_id="AC-DIFFERENT",  # Does not match profile
        profile_id="PROF-AERO-01",
        profile_version="1.0.0",
        source_reference="DOC",
    )
    with pytest.raises(LimitProfileMismatchError, match="vehicle_id 'AC-DIFFERENT' does not match"):
        EngineeringLimitProfile(
            limit_profile_id="LIM-PROF-1",
            vehicle_id="AC-AERO-X1",
            profile_id="PROF-AERO-01",
            profile_version="1.0.0",
            limits=(l_mismatch,),
        )


def test_vehicle_profile_compatibility_validation(
    sample_vehicle_profile: VehicleProfile,
) -> None:
    # Compatible profile
    p_valid = EngineeringLimitProfile(
        limit_profile_id="LIM-PROF-1",
        vehicle_id="AC-AERO-X1",
        profile_id="PROF-AERO-01",
        profile_version="1.0.0",
    )
    p_valid.validate_vehicle_profile_compatibility(sample_vehicle_profile)  # Does not raise

    # Mismatched vehicle_id
    p_bad_veh = EngineeringLimitProfile(
        limit_profile_id="LIM-PROF-1",
        vehicle_id="AC-OTHER",
        profile_id="PROF-AERO-01",
        profile_version="1.0.0",
    )
    with pytest.raises(LimitProfileMismatchError, match="vehicle_id 'AC-OTHER' does not match"):
        p_bad_veh.validate_vehicle_profile_compatibility(sample_vehicle_profile)

    # Mismatched profile_version
    p_bad_ver = EngineeringLimitProfile(
        limit_profile_id="LIM-PROF-1",
        vehicle_id="AC-AERO-X1",
        profile_id="PROF-AERO-01",
        profile_version="2.0.0",
    )
    with pytest.raises(
        LimitProfileMismatchError, match=r"profile_version '2\.0\.0' does not match"
    ):
        p_bad_ver.validate_vehicle_profile_compatibility(sample_vehicle_profile)


# ---------------------------------------------------------------------------
# 11. Malformed YAML, Missing Fields and Extra Fields
# ---------------------------------------------------------------------------


def test_load_malformed_yaml(limits_fixtures_dir: pathlib.Path) -> None:
    path = limits_fixtures_dir / "malformed_limit_profile.yaml"
    with pytest.raises(InvalidLimitProfileError, match="Malformed YAML syntax"):
        load_limit_profile_from_yaml(path)


def test_load_nonexistent_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "does_not_exist.yaml"
    with pytest.raises(InvalidLimitProfileError, match="Cannot read limit profile file"):
        load_limit_profile_from_yaml(path)


def test_load_missing_required_fields(limits_fixtures_dir: pathlib.Path) -> None:
    path = limits_fixtures_dir / "missing_required_limit_profile.yaml"
    with pytest.raises(InvalidLimitProfileError, match="Validation failed"):
        load_limit_profile_from_yaml(path)


def test_load_extra_forbidden_fields(limits_fixtures_dir: pathlib.Path) -> None:
    path = limits_fixtures_dir / "extra_fields_limit_profile.yaml"
    with pytest.raises(InvalidLimitProfileError, match="Extra inputs are not permitted"):
        load_limit_profile_from_yaml(path)


def test_load_duplicate_limit_ids(limits_fixtures_dir: pathlib.Path) -> None:
    path = limits_fixtures_dir / "duplicate_limit_id_profile.yaml"
    with pytest.raises(DuplicateLimitError, match="Duplicate limit_id found in profile"):
        load_limit_profile_from_yaml(path)


def test_load_conflicting_limits(limits_fixtures_dir: pathlib.Path) -> None:
    path = limits_fixtures_dir / "conflicting_limits_profile.yaml"
    with pytest.raises(ConflictingLimitError, match="Contradictory upper bounds"):
        load_limit_profile_from_yaml(path)


# ---------------------------------------------------------------------------
# 12. Atomic Failure Without Partial Results
# ---------------------------------------------------------------------------


def test_atomic_failure_returns_no_partial_profile(tmp_path: pathlib.Path) -> None:
    bad_yaml = """
engineering_limit_profile:
  limit_profile_id: LIM-ATOMIC-01
  vehicle_id: AC-01
  profile_id: PROF-01
  profile_version: "1.0.0"
  limits:
    - limit_id: LIM-VALID
      parameter_id: airspeed
      limit_type: UPPER_BOUND
      upper_bound: 250.0
      unit: kts
      vehicle_id: AC-01
      profile_id: PROF-01
      profile_version: "1.0.0"
      source_reference: "AFM"
    - limit_id: LIM-BAD
      parameter_id: airspeed
      limit_type: UPPER_BOUND
      upper_bound: -inf
      unit: kts
      vehicle_id: AC-01
      profile_id: PROF-01
      profile_version: "1.0.0"
      source_reference: "AFM"
"""
    file_path = tmp_path / "atomic_bad.yaml"
    file_path.write_text(bad_yaml, encoding="utf-8")

    result = None
    with pytest.raises(InvalidLimitProfileError):
        result = load_limit_profile_from_yaml(file_path)

    assert result is None


# ---------------------------------------------------------------------------
# 13. Deterministic Repeated Loading
# ---------------------------------------------------------------------------


def test_deterministic_repeated_loading(
    limits_fixtures_dir: pathlib.Path,
    sample_vehicle_profile: VehicleProfile,
    sample_registry: ParameterRegistry,
) -> None:
    valid_path = limits_fixtures_dir / "valid_limit_profile.yaml"

    prof1 = load_limit_profile_from_yaml(
        valid_path,
        expected_profile=sample_vehicle_profile,
        parameter_registry=sample_registry,
    )
    prof2 = load_engineering_limit_profile_from_yaml(
        valid_path,
        expected_profile=sample_vehicle_profile,
        parameter_registry=sample_registry,
    )

    assert prof1 == prof2
    assert prof1.model_dump() == prof2.model_dump()
    assert prof1.limit_profile_id == "LIM-PROF-AERO-01"
    assert len(prof1.limits) == 4


# ---------------------------------------------------------------------------
# 14. Compatibility with Step 13b Models and Epistemic Boundaries
# ---------------------------------------------------------------------------


def test_step_13b_compatibility_and_telemetry_rejection(
    sample_vehicle_profile: VehicleProfile,
) -> None:
    envelope = ConfigurationEnvelope(
        configuration_id="CONF-FLAPS-FULL",
        vehicle_id=sample_vehicle_profile.vehicle_id,
        profile_id=sample_vehicle_profile.profile_id,
        profile_version=sample_vehicle_profile.profile_version,
    )
    envelope.validate_profile_compatibility(sample_vehicle_profile)

    limit_flaps = ParameterLimitDefinition(
        limit_id="LIM-FLAPS",
        parameter_id="airspeed",
        limit_type=LimitType.UPPER_BOUND,
        upper_bound=140.0,
        unit="kts",
        vehicle_id=sample_vehicle_profile.vehicle_id,
        profile_id=sample_vehicle_profile.profile_id,
        profile_version=sample_vehicle_profile.profile_version,
        configuration_id=envelope.configuration_id,
        source_reference="AFM",
    )
    assert limit_flaps.configuration_id == envelope.configuration_id

    # Reject telemetry DataFrames
    df = pd.DataFrame({"airspeed": [100.0, 150.0]})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ParameterLimitDefinition(  # type: ignore[call-arg]
            limit_id="LIM-TELEMETRY",
            parameter_id="airspeed",
            limit_type=LimitType.UPPER_BOUND,
            upper_bound=250.0,
            unit="kts",
            vehicle_id="AC-1",
            profile_id="PROF-1",
            profile_version="1.0",
            source_reference="AFM",
            telemetry=df,
        )
