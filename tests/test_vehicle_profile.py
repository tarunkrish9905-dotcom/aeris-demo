"""
Comprehensive tests for VehicleProfile and ConfigurationEnvelope domain models,
atomic loaders, error handling, compatibility validation, and boundaries.

DISCLAIMER: All test vehicles and profiles in this module are synthetic and non-operational.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest
from pydantic import ValidationError

from aeris.core.registry import (
    ParameterDataType,
    ParameterDefinition,
    ParameterRegistry,
)
from aeris.core.vehicle import (
    ConfigurationEnvelope,
    InvalidConfigurationEnvelopeError,
    InvalidVehicleProfileError,
    ProfileMismatchError,
    VehicleProfile,
    load_configuration_envelope_from_yaml,
    load_vehicle_profile_from_yaml,
)


@pytest.fixture
def vehicle_fixtures_dir(fixtures_dir: pathlib.Path) -> pathlib.Path:
    """Return the vehicle fixtures directory."""
    return fixtures_dir / "vehicle"


# ---------------------------------------------------------------------------
# 1. Valid Profile Construction
# ---------------------------------------------------------------------------


def test_valid_minimal_vehicle_profile() -> None:
    profile = VehicleProfile(
        profile_id="PROF-001",
        vehicle_id="AC-100",
        profile_version="1.0.0",
        name="Test Profile",
    )
    assert profile.profile_id == "PROF-001"
    assert profile.vehicle_id == "AC-100"
    assert profile.profile_version == "1.0.0"
    assert profile.name == "Test Profile"
    assert profile.description is None
    assert profile.source_reference is None
    assert profile.supported_parameter_ids == ()
    assert profile.metadata == {}


def test_valid_fully_populated_vehicle_profile() -> None:
    profile = VehicleProfile(
        profile_id="PROF-002",
        vehicle_id="AC-200",
        profile_version="2.1.0-alpha",
        name="Full Profile",
        description="Detailed vehicle profile for test instrumentation.",
        source_reference="ECO-2026-889",
        supported_parameter_ids=("airspeed", "altitude", "roll_rate"),
        metadata={"synth_lead": "engineer_a", "tail_number": "N999XX"},
    )
    assert profile.profile_id == "PROF-002"
    assert profile.vehicle_id == "AC-200"
    assert profile.profile_version == "2.1.0-alpha"
    assert profile.name == "Full Profile"
    assert profile.description == "Detailed vehicle profile for test instrumentation."
    assert profile.source_reference == "ECO-2026-889"
    assert profile.supported_parameter_ids == ("airspeed", "altitude", "roll_rate")
    assert profile.metadata == {"synth_lead": "engineer_a", "tail_number": "N999XX"}


# ---------------------------------------------------------------------------
# 2. Missing and Malformed Required Identifiers
# ---------------------------------------------------------------------------


def test_missing_profile_identifiers() -> None:
    with pytest.raises(ValidationError) as exc:
        VehicleProfile(  # type: ignore[call-arg]
            vehicle_id="AC-100",
            profile_version="1.0.0",
            name="Missing ID",
        )
    assert "profile_id" in str(exc.value)

    with pytest.raises(ValidationError) as exc:
        VehicleProfile(  # type: ignore[call-arg]
            profile_id="PROF-001",
            profile_version="1.0.0",
            name="Missing Vehicle ID",
        )
    assert "vehicle_id" in str(exc.value)


def test_empty_or_whitespace_profile_identifiers() -> None:
    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        VehicleProfile(
            profile_id="",
            vehicle_id="AC-100",
            profile_version="1.0.0",
            name="Name",
        )

    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        VehicleProfile(
            profile_id="   ",
            vehicle_id="AC-100",
            profile_version="1.0.0",
            name="Name",
        )

    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        VehicleProfile(
            profile_id="PROF-1",
            vehicle_id="",
            profile_version="1.0.0",
            name="Name",
        )

    with pytest.raises(ValidationError, match="cannot be empty or solely whitespace"):
        VehicleProfile(
            profile_id="PROF-1",
            vehicle_id="AC-1",
            profile_version="1.0.0",
            name="   ",
        )


def test_malformed_profile_identifiers() -> None:
    with pytest.raises(ValidationError, match="alphanumeric characters, dashes, and underscores"):
        VehicleProfile(
            profile_id="PROF 001",  # Space
            vehicle_id="AC-100",
            profile_version="1.0.0",
            name="Name",
        )

    with pytest.raises(ValidationError, match="alphanumeric characters, dashes, and underscores"):
        VehicleProfile(
            profile_id="PROF@001",  # Special character
            vehicle_id="AC-100",
            profile_version="1.0.0",
            name="Name",
        )


# ---------------------------------------------------------------------------
# 3. Profile Version and Revision Validation
# ---------------------------------------------------------------------------


def test_profile_version_formatting() -> None:
    # Valid versions
    for valid_ver in ("1.0", "1.0.0", "v1.2.3", "2026.1", "2.0-rc1"):
        p = VehicleProfile(
            profile_id="P-1",
            vehicle_id="AC-1",
            profile_version=valid_ver,
            name="Name",
        )
        assert p.profile_version == valid_ver

    # Invalid versions
    for invalid_ver in ("", "   ", "singleword", "1", "version1"):
        with pytest.raises(ValidationError, match="profile_version"):
            VehicleProfile(
                profile_id="P-1",
                vehicle_id="AC-1",
                profile_version=invalid_ver,
                name="Name",
            )


# ---------------------------------------------------------------------------
# 4. Supported Parameter References and Unknown Parameters
# ---------------------------------------------------------------------------


def test_supported_parameters_duplicates_and_clean_ids() -> None:
    # Duplicate parameter IDs in profile
    with pytest.raises(ValidationError, match="Duplicate parameter_id found"):
        VehicleProfile(
            profile_id="P-1",
            vehicle_id="AC-1",
            profile_version="1.0",
            name="Name",
            supported_parameter_ids=("airspeed", "altitude", "airspeed"),
        )

    # Malformed parameter ID
    with pytest.raises(ValidationError, match="alphanumeric characters, dashes, and underscores"):
        VehicleProfile(
            profile_id="P-1",
            vehicle_id="AC-1",
            profile_version="1.0",
            name="Name",
            supported_parameter_ids=("airspeed", "invalid param name"),
        )


def test_loader_cross_references_unknown_parameters(tmp_path: pathlib.Path) -> None:
    registry = ParameterRegistry()
    registry.register(
        ParameterDefinition(
            parameter_id="airspeed",
            display_name="Airspeed",
            description="Airspeed parameter",
            physical_quantity="velocity",
            data_type=ParameterDataType.FLOAT,
        )
    )

    yaml_content = """
vehicle_profile:
  profile_id: PROF-UNKNOWN-PARAM
  vehicle_id: AC-01
  profile_version: "1.0.0"
  name: "Profile with unknown params"
  supported_parameter_ids:
    - airspeed
    - nonexistent_engine_temp
"""
    yaml_file = tmp_path / "unknown_param.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    # With parameter registry, must raise InvalidVehicleProfileError
    with pytest.raises(InvalidVehicleProfileError, match="references unknown parameter IDs"):
        load_vehicle_profile_from_yaml(yaml_file, parameter_registry=registry)

    # Without parameter registry, loads successfully
    p = load_vehicle_profile_from_yaml(yaml_file, parameter_registry=None)
    assert "nonexistent_engine_temp" in p.supported_parameter_ids


# ---------------------------------------------------------------------------
# 5. Valid Configuration-Envelope Construction
# ---------------------------------------------------------------------------


def test_valid_configuration_envelope() -> None:
    env = ConfigurationEnvelope(
        configuration_id="CONF-FLAPS-20",
        vehicle_id="AC-AERO-X1",
        profile_id="PROF-AERO-01",
        profile_version="1.0.0",
        configuration_version="1.1.0",
        name="Flaps 20 Config",
        description="Takeoff flap setting envelope.",
        source_reference="CFG-DOC-02",
        metadata={"flaps_deg": 20.0},
    )
    assert env.configuration_id == "CONF-FLAPS-20"
    assert env.vehicle_id == "AC-AERO-X1"
    assert env.profile_id == "PROF-AERO-01"
    assert env.profile_version == "1.0.0"
    assert env.configuration_version == "1.1.0"
    assert env.name == "Flaps 20 Config"
    assert env.metadata["flaps_deg"] == 20.0


# ---------------------------------------------------------------------------
# 6. Configuration/Profile Identity and Version Mismatch
# ---------------------------------------------------------------------------


def test_envelope_compatibility_validation() -> None:
    profile = VehicleProfile(
        profile_id="PROF-AERO-01",
        vehicle_id="AC-AERO-X1",
        profile_version="1.0.0",
        name="Profile",
    )

    # Compatible envelope
    valid_env = ConfigurationEnvelope(
        configuration_id="CONF-01",
        vehicle_id="AC-AERO-X1",
        profile_id="PROF-AERO-01",
        profile_version="1.0.0",
    )
    valid_env.validate_profile_compatibility(profile)  # Does not raise

    # Mismatched vehicle_id
    bad_vehicle_env = ConfigurationEnvelope(
        configuration_id="CONF-01",
        vehicle_id="AC-OTHER-X2",
        profile_id="PROF-AERO-01",
        profile_version="1.0.0",
    )
    with pytest.raises(ProfileMismatchError, match="vehicle_id 'AC-OTHER-X2' does not match"):
        bad_vehicle_env.validate_profile_compatibility(profile)

    # Mismatched profile_id
    bad_profile_env = ConfigurationEnvelope(
        configuration_id="CONF-01",
        vehicle_id="AC-AERO-X1",
        profile_id="PROF-OTHER-99",
        profile_version="1.0.0",
    )
    with pytest.raises(ProfileMismatchError, match="profile_id 'PROF-OTHER-99' does not match"):
        bad_profile_env.validate_profile_compatibility(profile)

    # Mismatched profile_version
    bad_version_env = ConfigurationEnvelope(
        configuration_id="CONF-01",
        vehicle_id="AC-AERO-X1",
        profile_id="PROF-AERO-01",
        profile_version="2.0.0",
    )
    with pytest.raises(ProfileMismatchError, match=r"profile_version '2\.0\.0' does not match"):
        bad_version_env.validate_profile_compatibility(profile)


# ---------------------------------------------------------------------------
# 7. Missing or Malformed YAML
# ---------------------------------------------------------------------------


def test_load_malformed_yaml(vehicle_fixtures_dir: pathlib.Path) -> None:
    malformed_path = vehicle_fixtures_dir / "malformed_profile.yaml"
    with pytest.raises(InvalidVehicleProfileError, match="Malformed YAML syntax"):
        load_vehicle_profile_from_yaml(malformed_path)

    with pytest.raises(InvalidConfigurationEnvelopeError, match="Malformed YAML syntax"):
        load_configuration_envelope_from_yaml(malformed_path)


def test_load_nonexistent_file(tmp_path: pathlib.Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(InvalidVehicleProfileError, match="Cannot read vehicle profile file"):
        load_vehicle_profile_from_yaml(missing)

    with pytest.raises(
        InvalidConfigurationEnvelopeError, match="Cannot read configuration envelope file"
    ):
        load_configuration_envelope_from_yaml(missing)


# ---------------------------------------------------------------------------
# 8. Invalid Fields and Inconsistent Content
# ---------------------------------------------------------------------------


def test_load_missing_required_fields(vehicle_fixtures_dir: pathlib.Path) -> None:
    missing_path = vehicle_fixtures_dir / "missing_required_profile.yaml"
    with pytest.raises(InvalidVehicleProfileError, match="Validation failed for vehicle profile"):
        load_vehicle_profile_from_yaml(missing_path)


def test_load_extra_forbidden_fields(vehicle_fixtures_dir: pathlib.Path) -> None:
    extra_path = vehicle_fixtures_dir / "extra_fields_profile.yaml"
    with pytest.raises(InvalidVehicleProfileError, match="Extra inputs are not permitted"):
        load_vehicle_profile_from_yaml(extra_path)


# ---------------------------------------------------------------------------
# 9. Atomic Failure: No Partial Profile Returned
# ---------------------------------------------------------------------------


def test_atomic_failure_does_not_yield_partial_object(tmp_path: pathlib.Path) -> None:
    bad_yaml = """
vehicle_profile:
  profile_id: PROF-ATOMIC-01
  vehicle_id: AC-01
  profile_version: "invalid_not_dotted"
  name: "Atomic Test"
"""
    file_path = tmp_path / "atomic_bad.yaml"
    file_path.write_text(bad_yaml, encoding="utf-8")

    result = None
    with pytest.raises(InvalidVehicleProfileError):
        result = load_vehicle_profile_from_yaml(file_path)

    assert result is None


# ---------------------------------------------------------------------------
# 10. Deterministic Repeated Loading
# ---------------------------------------------------------------------------


def test_deterministic_repeated_loading(vehicle_fixtures_dir: pathlib.Path) -> None:
    valid_path = vehicle_fixtures_dir / "valid_vehicle_profile.yaml"
    p1 = load_vehicle_profile_from_yaml(valid_path)
    p2 = load_vehicle_profile_from_yaml(valid_path)

    assert p1 == p2
    assert p1.model_dump() == p2.model_dump()
    assert p1.profile_id == "PROF-AERO-01"


def test_deterministic_repeated_envelope_loading(vehicle_fixtures_dir: pathlib.Path) -> None:
    valid_env_path = vehicle_fixtures_dir / "valid_configuration_envelope.yaml"
    profile_path = vehicle_fixtures_dir / "valid_vehicle_profile.yaml"
    profile = load_vehicle_profile_from_yaml(profile_path)

    env1 = load_configuration_envelope_from_yaml(valid_env_path, expected_profile=profile)
    env2 = load_configuration_envelope_from_yaml(valid_env_path, expected_profile=profile)

    assert env1 == env2
    assert env1.model_dump() == env2.model_dump()
    assert env1.configuration_id == "CONF-CLEAN-01"


# ---------------------------------------------------------------------------
# 11. Regression Tests for Existing Vehicle / Configuration / Context Behavior
# ---------------------------------------------------------------------------


def test_existing_vehicle_and_configuration_models_remain_untouched() -> None:
    from aeris.core.vehicle.configuration import VehicleConfiguration
    from aeris.core.vehicle.models import Vehicle
    from aeris.core.vehicle.test_context import TestContext

    v = Vehicle(vehicle_id="AC-ORIG-01", name="Original")
    c = VehicleConfiguration(configuration_id="CONF-ORIG-01", vehicle_id="AC-ORIG-01")
    ctx = TestContext(
        context_id="CTX-ORIG-01", vehicle_id="AC-ORIG-01", configuration_id="CONF-ORIG-01"
    )

    assert v.vehicle_id == "AC-ORIG-01"
    assert c.configuration_id == "CONF-ORIG-01"
    assert ctx.context_id == "CTX-ORIG-01"

    # Strict isolation from performance limits still holds
    with pytest.raises(ValidationError):
        Vehicle(vehicle_id="AC-1", max_speed_kts=300)  # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        VehicleConfiguration(configuration_id="C-1", stall_speed=60)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 12. Negative Tests: No Telemetry or DataFrame Attachment
# ---------------------------------------------------------------------------


def test_profile_rejects_telemetry_and_limits() -> None:
    df = pd.DataFrame({"airspeed": [100.0, 110.0]})

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        VehicleProfile(  # type: ignore[call-arg]
            profile_id="P-1",
            vehicle_id="AC-1",
            profile_version="1.0.0",
            name="P",
            telemetry=df,
        )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        VehicleProfile(  # type: ignore[call-arg]
            profile_id="P-1",
            vehicle_id="AC-1",
            profile_version="1.0.0",
            name="P",
            v_ne_kts=250.0,
        )


def test_envelope_rejects_telemetry_and_limits() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ConfigurationEnvelope(  # type: ignore[call-arg]
            configuration_id="C-1",
            vehicle_id="AC-1",
            profile_id="P-1",
            profile_version="1.0.0",
            max_thrust_lbs=5000.0,
        )
