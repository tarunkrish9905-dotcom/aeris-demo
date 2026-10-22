"""
AERIS Core Quality package.
"""

from aeris.core.quality.quality_engine import (
    DEFAULT_SCORING_WEIGHTS,
    DataQualityEngine,
    assess_data_quality,
)

__all__ = [
    "DEFAULT_SCORING_WEIGHTS",
    "DataQualityEngine",
    "assess_data_quality",
]
