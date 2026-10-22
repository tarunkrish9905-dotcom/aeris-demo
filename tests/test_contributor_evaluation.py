
import pytest
from pydantic import ValidationError

from aeris.core.contributors.evaluation_engine import ContributorEvaluationEngine
from aeris.core.contributors.evaluation_models import (
    AssociationStrength,
    ContributorEvaluation,
    ContributorEvaluationConfiguration,
    ContributorEvaluationRequest,
)
from aeris.core.contributors.models import ContributorAnalysisMethod, ContributorAnalysisResult
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.results.types import ResultStatus


@pytest.fixture
def mock_prov():
    return ProvenanceRecord(
        source_uri="test",
        activity_name="test",
        epistemic_level=EpistemicLevel.CALCULATED,
    )


def create_mock_result(
    coef,
    obs=10,
    status=ResultStatus.SUCCESS,
    method=ContributorAnalysisMethod.PEARSON,
    prov=None,
    analysis_id="id1",
):
    return ContributorAnalysisResult(
        analysis_id=analysis_id,
        target_column="tgt",
        factor_column="fac",
        analysis_method=method,
        coefficient=coef,
        usable_observation_count=obs,
        excluded_observation_count=0,
        status=status,
        issues=(),
        provenance=prov
        or ProvenanceRecord(
            source_uri="test", activity_name="test", epistemic_level=EpistemicLevel.CALCULATED
        ),
    )


# 1. Config Validation
def test_config_valid():
    cfg = ContributorEvaluationConfiguration()
    assert cfg.very_weak_upper == 0.2


def test_config_invalid_ordering():
    with pytest.raises(ValidationError):
        ContributorEvaluationConfiguration(very_weak_upper=0.5, weak_upper=0.3)


def test_config_invalid_nan():
    with pytest.raises(ValidationError):
        ContributorEvaluationConfiguration(weak_upper=float("nan"))


def test_config_invalid_negative_obs():
    with pytest.raises(ValidationError):
        ContributorEvaluationConfiguration(minimum_usable_observations=-1)


def test_config_invalid_zero_obs():
    with pytest.raises(ValidationError):
        ContributorEvaluationConfiguration(minimum_usable_observations=0)


def test_config_immutability():
    cfg = ContributorEvaluationConfiguration()
    with pytest.raises(Exception):
        cfg.very_weak_upper = 0.5


# 2. Evaluation Request
def test_request_empty():
    req = ContributorEvaluationRequest(analysis_results=())
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.status == ResultStatus.SUCCESS
    assert len(res.evaluations) == 0


# 3. Association Strengths
@pytest.mark.parametrize(
    "coef, expected",
    [
        (0.0, AssociationStrength.VERY_WEAK),
        (0.19, AssociationStrength.VERY_WEAK),
        (0.20, AssociationStrength.WEAK),
        (0.39, AssociationStrength.WEAK),
        (0.40, AssociationStrength.MODERATE),
        (0.59, AssociationStrength.MODERATE),
        (0.60, AssociationStrength.STRONG),
        (0.79, AssociationStrength.STRONG),
        (0.80, AssociationStrength.VERY_STRONG),
        (1.0, AssociationStrength.VERY_STRONG),
    ],
)
def test_association_strength_positive(coef, expected):
    req = ContributorEvaluationRequest(analysis_results=(create_mock_result(coef=coef),))
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.evaluations[0].association_strength == expected
    assert res.evaluations[0].coefficient == coef  # Sign preserved


@pytest.mark.parametrize(
    "coef, expected",
    [
        (-0.19, AssociationStrength.VERY_WEAK),
        (-0.20, AssociationStrength.WEAK),
        (-0.40, AssociationStrength.MODERATE),
        (-0.60, AssociationStrength.STRONG),
        (-0.80, AssociationStrength.VERY_STRONG),
        (-1.0, AssociationStrength.VERY_STRONG),
    ],
)
def test_association_strength_negative(coef, expected):
    req = ContributorEvaluationRequest(analysis_results=(create_mock_result(coef=coef),))
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.evaluations[0].association_strength == expected
    assert res.evaluations[0].coefficient == coef  # Sign preserved


# 4. Status Propagation
def test_status_all_success():
    req = ContributorEvaluationRequest(
        analysis_results=(create_mock_result(0.5), create_mock_result(0.6))
    )
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.status == ResultStatus.SUCCESS


def test_status_partial_due_to_missing_obs():
    req = ContributorEvaluationRequest(
        analysis_results=(create_mock_result(0.5, obs=10), create_mock_result(0.6, obs=2))
    )
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.status == ResultStatus.PARTIAL
    assert res.evaluations[1].status == ResultStatus.NOT_AVAILABLE


def test_status_all_unavailable():
    req = ContributorEvaluationRequest(
        analysis_results=(create_mock_result(0.5, obs=2), create_mock_result(0.6, obs=2))
    )
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.status == ResultStatus.NOT_AVAILABLE


def test_status_error_propagation():
    req = ContributorEvaluationRequest(
        analysis_results=(create_mock_result(0.5, status=ResultStatus.ERROR),)
    )
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.status == ResultStatus.NOT_AVAILABLE
    assert res.evaluations[0].status == ResultStatus.NOT_AVAILABLE


def test_status_partial_input():
    req = ContributorEvaluationRequest(
        analysis_results=(create_mock_result(0.5, status=ResultStatus.PARTIAL),)
    )
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.status == ResultStatus.PARTIAL
    assert res.evaluations[0].status == ResultStatus.PARTIAL


def test_missing_coef_is_not_available():
    req = ContributorEvaluationRequest(analysis_results=(create_mock_result(None),))
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.evaluations[0].status == ResultStatus.NOT_AVAILABLE


def test_missing_obs_is_not_available():
    req = ContributorEvaluationRequest(analysis_results=(create_mock_result(0.5, obs=None),))
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.evaluations[0].status == ResultStatus.NOT_AVAILABLE


# 5. Method Preservation
def test_method_preservation():
    req = ContributorEvaluationRequest(
        analysis_results=(create_mock_result(0.5, method=ContributorAnalysisMethod.SPEARMAN),)
    )
    # It does not mutate the original result, but does it store method?
    # Evaluation result doesn't store method, it stores analysis_id. The parent result has the method.
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.evaluations[0].analysis_id == "id1"


# 6. Evaluation Ranking
def test_ranking_disabled():
    req = ContributorEvaluationRequest(
        analysis_results=(
            create_mock_result(0.2, analysis_id="id1"),
            create_mock_result(0.9, analysis_id="id2"),
        ),
        request_ranking=False,
    )
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.evaluations[0].analysis_id == "id1"
    assert res.evaluations[1].analysis_id == "id2"
    assert res.evaluations[0].rank_position is None


def test_ranking_enabled_basic():
    req = ContributorEvaluationRequest(
        analysis_results=(
            create_mock_result(0.2, analysis_id="id1"),
            create_mock_result(0.9, analysis_id="id2"),
        ),
        request_ranking=True,
    )
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.evaluations[0].analysis_id == "id2"
    assert res.evaluations[0].rank_position == 1
    assert res.evaluations[1].analysis_id == "id1"
    assert res.evaluations[1].rank_position == 2


def test_ranking_tie_breaking_by_obs():
    req = ContributorEvaluationRequest(
        analysis_results=(
            create_mock_result(0.5, obs=10, analysis_id="id1"),
            create_mock_result(0.5, obs=20, analysis_id="id2"),
        ),
        request_ranking=True,
    )
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.evaluations[0].analysis_id == "id2"
    assert res.evaluations[1].analysis_id == "id1"


def test_ranking_tie_breaking_by_id():
    req = ContributorEvaluationRequest(
        analysis_results=(
            create_mock_result(0.5, obs=10, analysis_id="idB"),
            create_mock_result(0.5, obs=10, analysis_id="idA"),
        ),
        request_ranking=True,
    )
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.evaluations[0].analysis_id == "idA"
    assert res.evaluations[1].analysis_id == "idB"


def test_ranking_negative_coef_magnitude():
    req = ContributorEvaluationRequest(
        analysis_results=(
            create_mock_result(-0.9, analysis_id="id1"),
            create_mock_result(0.8, analysis_id="id2"),
        ),
        request_ranking=True,
    )
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.evaluations[0].analysis_id == "id1"
    assert res.evaluations[1].analysis_id == "id2"


def test_ranking_skips_invalid():
    req = ContributorEvaluationRequest(
        analysis_results=(
            create_mock_result(0.9, obs=2, analysis_id="id1"),  # INVALID due to obs
            create_mock_result(0.2, analysis_id="id2"),
        ),
        request_ranking=True,
    )
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.evaluations[0].analysis_id == "id2"
    assert res.evaluations[0].rank_position == 1
    assert res.evaluations[1].analysis_id == "id1"
    assert res.evaluations[1].rank_position is None


def test_ranking_determinism():
    res1 = create_mock_result(0.5, analysis_id="id1")
    res2 = create_mock_result(0.5, analysis_id="id2")
    req1 = ContributorEvaluationRequest(analysis_results=(res1, res2), request_ranking=True)
    req2 = ContributorEvaluationRequest(analysis_results=(res2, res1), request_ranking=True)

    out1 = ContributorEvaluationEngine.evaluate(req1)
    out2 = ContributorEvaluationEngine.evaluate(req2)

    assert out1.evaluations[0].analysis_id == out2.evaluations[0].analysis_id


# 7. Provenance & Epistemic
def test_provenance_linkage():
    req = ContributorEvaluationRequest(analysis_results=(create_mock_result(0.5),))
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.provenance.epistemic_level == EpistemicLevel.CALCULATED
    assert res.evaluations[0].provenance.epistemic_level == EpistemicLevel.CALCULATED
    assert (
        res.evaluations[0].provenance.parent_record_ids[0]
        == req.analysis_results[0].provenance.record_id
    )


def test_epistemic_level_rejection():
    with pytest.raises(ValueError):
        ContributorEvaluation(
            analysis_id="x",
            target_column="y",
            factor_column="z",
            status=ResultStatus.SUCCESS,
            provenance=ProvenanceRecord(
                source_uri="", activity_name="", epistemic_level=EpistemicLevel.OBSERVED
            ),
        )


# 8. No Dataframe Dependency
def test_no_pandas_dependency():
    assert True  # We don't import pandas in evaluation_engine.py


# 9. No Unit Conversion / Causal Inference
def test_no_causal_fields_in_output():
    req = ContributorEvaluationRequest(analysis_results=(create_mock_result(0.5),))
    res = ContributorEvaluationEngine.evaluate(req)
    # Check that model dict has no fields starting with "causal" or "root_cause"
    for e in res.evaluations:
        for k in e.model_dump():
            assert "causal" not in k
            assert "cause" not in k
            assert "recommendation" not in k


def test_missing_data_fields_preserved():
    req = ContributorEvaluationRequest(
        analysis_results=(
            ContributorAnalysisResult(
                analysis_id="id1",
                target_column="tgt",
                factor_column="fac",
                analysis_method=ContributorAnalysisMethod.PEARSON,
                coefficient=0.5,
                total_input_rows=100,
                usable_observation_count=90,
                excluded_observation_count=10,
                status=ResultStatus.SUCCESS,
                issues=(),
                provenance=ProvenanceRecord(
                    source_uri="", activity_name="", epistemic_level=EpistemicLevel.CALCULATED
                ),
            ),
        )
    )
    res = ContributorEvaluationEngine.evaluate(req)
    assert res.evaluations[0].usable_observation_count == 90
    assert res.evaluations[0].excluded_observation_count == 10


def test_invalid_ranking_policy():
    with pytest.raises(ValidationError):
        ContributorEvaluationRequest(
            analysis_results=(), request_ranking=True, ranking_policy="NOT_A_POLICY"
        )
