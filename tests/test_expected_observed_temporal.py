import datetime

import pytest

from aeris.core.expected_observed.engine import TemporalExpectedObservedEngine
from aeris.core.expected_observed.models import (
    ComparisonMethod,
    ExpectationSource,
    TemporalExpectedObservedRequest,
    TemporalExpectedValue,
    TemporalObservedValue,
    Tolerance,
    ToleranceType,
)
from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import ResultStatus


def dt(hours=0, minutes=0, seconds=0, tz=datetime.UTC) -> datetime.datetime:
    return datetime.datetime(2026, 1, 1, hours, minutes, seconds, tzinfo=tz)


def create_e(
    val: float,
    t: datetime.datetime,
    param: str = "airspeed",
    unit: str | None = "kt",
    id: str = "e1",
) -> TemporalExpectedValue:
    return TemporalExpectedValue(
        expected_id=id,
        parameter_id=param,
        value=val,
        unit=unit,
        source=ExpectationSource.EXPLICIT_INPUT,
        source_reference="test",
        epistemic_level=EpistemicLevel.PROCEDURAL,
        timestamp=t,
    )


def create_o(
    val: float,
    t: datetime.datetime,
    param: str = "airspeed",
    unit: str | None = "kt",
    id: str = "o1",
) -> TemporalObservedValue:
    return TemporalObservedValue(
        observed_id=id,
        parameter_id=param,
        value=val,
        unit=unit,
        source_reference="test_obs",
        timestamp=t,
    )


def test_temporal_exact_match():
    e1 = create_e(100.0, dt(0))
    o1 = create_o(105.0, dt(0))

    req = TemporalExpectedObservedRequest(
        comparison_id="c1",
        parameter_id="airspeed",
        expected_values=(e1,),
        observed_values=(o1,),
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    res = TemporalExpectedObservedEngine.compare_temporal(req)
    assert res.status == ResultStatus.SUCCESS
    assert res.matched_count == 1
    assert res.expected_only_count == 0
    assert res.observed_only_count == 0
    assert res.matched_points[0].comparison.deviation.deviation_value == 5.0


def test_temporal_no_matches():
    e1 = create_e(100.0, dt(0))
    o1 = create_o(105.0, dt(1))

    req = TemporalExpectedObservedRequest(
        comparison_id="c1",
        parameter_id="airspeed",
        expected_values=(e1,),
        observed_values=(o1,),
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    res = TemporalExpectedObservedEngine.compare_temporal(req)
    assert res.status == ResultStatus.NOT_AVAILABLE
    assert res.matched_count == 0
    assert res.expected_only_count == 1
    assert res.observed_only_count == 1


def test_temporal_partial_coverage():
    e_vals = (create_e(100.0, dt(0)), create_e(200.0, dt(1)), create_e(300.0, dt(2)))
    o_vals = (create_o(105.0, dt(0)), create_o(320.0, dt(2)), create_o(400.0, dt(3)))

    req = TemporalExpectedObservedRequest(
        comparison_id="c1",
        parameter_id="airspeed",
        expected_values=e_vals,
        observed_values=o_vals,
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    res = TemporalExpectedObservedEngine.compare_temporal(req)
    assert res.status == ResultStatus.PARTIAL
    assert res.matched_count == 2
    assert res.expected_only_count == 1
    assert res.observed_only_count == 1


def test_temporal_naive_timestamp_rejection():
    with pytest.raises(ValueError, match="timezone-aware"):
        create_e(100.0, datetime.datetime(2026, 1, 1))


def test_temporal_non_utc_timestamp_rejection():
    tz_ny = datetime.timezone(datetime.timedelta(hours=-5))
    with pytest.raises(ValueError, match="UTC"):
        create_e(100.0, dt(0, tz=tz_ny))


def test_temporal_duplicate_expected_timestamp():
    e1 = create_e(100.0, dt(0), id="e1")
    e2 = create_e(200.0, dt(0), id="e2")
    req = TemporalExpectedObservedRequest(
        comparison_id="c1",
        parameter_id="airspeed",
        expected_values=(e1, e2),
        observed_values=(),
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    res = TemporalExpectedObservedEngine.compare_temporal(req)
    assert res.status == ResultStatus.ERROR
    assert "ERR_DUPLICATE_EXPECTED_TIMESTAMPS" in [i.issue_code for i in res.issues]


def test_temporal_duplicate_observed_timestamp():
    o1 = create_o(100.0, dt(0), id="o1")
    o2 = create_o(200.0, dt(0), id="o2")
    req = TemporalExpectedObservedRequest(
        comparison_id="c1",
        parameter_id="airspeed",
        expected_values=(),
        observed_values=(o1, o2),
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    res = TemporalExpectedObservedEngine.compare_temporal(req)
    assert res.status == ResultStatus.ERROR
    assert "ERR_DUPLICATE_OBSERVED_TIMESTAMPS" in [i.issue_code for i in res.issues]


def test_temporal_no_nearest_match():
    e1 = create_e(100.0, dt(0, 0, 0))
    o1 = create_o(105.0, dt(0, 0, 1))
    req = TemporalExpectedObservedRequest(
        comparison_id="c1",
        parameter_id="airspeed",
        expected_values=(e1,),
        observed_values=(o1,),
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    res = TemporalExpectedObservedEngine.compare_temporal(req)
    assert res.matched_count == 0


def test_temporal_ordering_semantics():
    e_vals = (create_e(200.0, dt(1)), create_e(100.0, dt(0)))
    o_vals = (create_o(105.0, dt(0)), create_o(190.0, dt(1)))
    req = TemporalExpectedObservedRequest(
        comparison_id="c1",
        parameter_id="airspeed",
        expected_values=e_vals,
        observed_values=o_vals,
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    res = TemporalExpectedObservedEngine.compare_temporal(req)
    assert res.matched_points[0].timestamp == dt(0)
    assert res.matched_points[1].timestamp == dt(1)


def test_temporal_zero_denominator():
    e_vals = (create_e(100.0, dt(0)), create_e(0.0, dt(1)), create_e(200.0, dt(2)))
    o_vals = (create_o(105.0, dt(0)), create_o(5.0, dt(1)), create_o(210.0, dt(2)))
    req = TemporalExpectedObservedRequest(
        comparison_id="c1",
        parameter_id="airspeed",
        expected_values=e_vals,
        observed_values=o_vals,
        method=ComparisonMethod.PERCENT_DIFFERENCE,
    )
    res = TemporalExpectedObservedEngine.compare_temporal(req)
    assert res.status == ResultStatus.PARTIAL
    assert res.matched_points[0].comparison.status == ResultStatus.SUCCESS
    assert res.matched_points[1].comparison.status == ResultStatus.ERROR
    assert res.matched_points[2].comparison.status == ResultStatus.SUCCESS


def test_temporal_tolerance():
    e_vals = (create_e(100.0, dt(0)), create_e(200.0, dt(1)), create_e(300.0, dt(2)))
    o_vals = (create_o(105.0, dt(0)), create_o(190.0, dt(1)), create_o(320.0, dt(2)))
    req = TemporalExpectedObservedRequest(
        comparison_id="c1",
        parameter_id="airspeed",
        expected_values=e_vals,
        observed_values=o_vals,
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
        tolerance=Tolerance(
            tolerance_type=ToleranceType.ABSOLUTE, lower_bound=-10.0, upper_bound=10.0
        ),
    )
    res = TemporalExpectedObservedEngine.compare_temporal(req)
    assert res.matched_points[0].comparison.within_tolerance is True
    assert res.matched_points[1].comparison.within_tolerance is True
    assert res.matched_points[2].comparison.within_tolerance is False


def test_temporal_parameter_identity_match():
    with pytest.raises(ValueError, match="parameter_id"):
        TemporalExpectedObservedRequest(
            comparison_id="c1",
            parameter_id="airspeed",
            expected_values=(create_e(100.0, dt(0), param="altitude"),),
            observed_values=(),
            method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
        )


def test_temporal_empty_inputs():
    req = TemporalExpectedObservedRequest(
        comparison_id="c1",
        parameter_id="airspeed",
        expected_values=(),
        observed_values=(),
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    res = TemporalExpectedObservedEngine.compare_temporal(req)
    assert res.status == ResultStatus.NOT_AVAILABLE
    assert res.matched_count == 0


def test_temporal_epistemic_level():
    req = TemporalExpectedObservedRequest(
        comparison_id="c1",
        parameter_id="airspeed",
        expected_values=(create_e(100.0, dt(0)),),
        observed_values=(create_o(105.0, dt(0)),),
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    res = TemporalExpectedObservedEngine.compare_temporal(req)
    assert res.provenance.epistemic_level == EpistemicLevel.CALCULATED
    assert res.matched_points[0].comparison.provenance.epistemic_level == EpistemicLevel.CALCULATED
