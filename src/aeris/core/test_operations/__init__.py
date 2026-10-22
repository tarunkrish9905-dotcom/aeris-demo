"""
Test Operations domain models.

This module provides explicit definitions for test points and flight
operations, representing *what* was planned or defined, completely
independent of *what* was measured or evaluated.
"""

from aeris.core.test_operations.association import (
    TestPointAssociationEngine,
    TestPointAssociationResult,
)
from aeris.core.test_operations.test_card import TestCard
from aeris.core.test_operations.test_point import TestPoint
from aeris.core.test_operations.test_point_observation import TestPointObservation

__all__ = [
    "TestCard",
    "TestPoint",
    "TestPointAssociationEngine",
    "TestPointAssociationResult",
    "TestPointObservation",
]
