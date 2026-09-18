"""Unit tests for strict Pydantic V2 schemas (Chapter 1)."""

import pytest
from pydantic import ValidationError

from app.schemas.prediction import (
    FEATURE_ORDER,
    PredictionRequest,
    TabularFeatures,
)


def test_valid_tabular_features(valid_features_dict: dict) -> None:
    features = TabularFeatures(**valid_features_dict)
    assert features.customer_tenure_months == 12
    assert features.has_dependents is True
    assert len(features.to_ordered_tuple()) == len(FEATURE_ORDER)


def test_strict_type_enforcement(valid_features_dict: dict) -> None:
    """Verifies that strict=True rejects implicit string-to-numeric coercion."""
    invalid = valid_features_dict.copy()
    invalid["customer_tenure_months"] = "12"  # String disallowed by strict int
    with pytest.raises(ValidationError):
        TabularFeatures(**invalid)


def test_extra_fields_forbidden(valid_features_dict: dict) -> None:
    """Verifies that extra='forbid' rejects undeclared payload fields."""
    invalid = valid_features_dict.copy()
    invalid["uncontracted_extra_field"] = "malicious_or_buggy_data"
    with pytest.raises(ValidationError):
        TabularFeatures(**invalid)


def test_business_invariant_charges(valid_features_dict: dict) -> None:
    """Verifies that total_charges cannot be less than monthly_charges if tenure >= 1."""
    invalid = valid_features_dict.copy()
    invalid["customer_tenure_months"] = 6
    invalid["monthly_charges"] = 100.0
    invalid["total_charges"] = 50.0  # Less than monthly_charges
    with pytest.raises(ValidationError) as exc:
        TabularFeatures(**invalid)
    assert "total_charges cannot be less than monthly_charges" in str(exc.value)


def test_reject_nil_uuid(valid_tabular_features: TabularFeatures) -> None:
    """Verifies that the nil UUID is explicitly rejected."""
    with pytest.raises(ValidationError):
        PredictionRequest(
            request_id="00000000-0000-0000-0000-000000000000",
            features=valid_tabular_features,
        )
