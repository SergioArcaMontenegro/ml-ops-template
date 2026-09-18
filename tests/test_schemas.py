"""Tests unitarios para validación estricta de esquemas Pydantic V2 (Capítulo 1)."""

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
    """Verifica que strict=True rechaza coerción implícita de strings a numéricos."""
    invalid = valid_features_dict.copy()
    invalid["customer_tenure_months"] = "12"  # String no permitido en int estricto
    with pytest.raises(ValidationError):
        TabularFeatures(**invalid)


def test_extra_fields_forbidden(valid_features_dict: dict) -> None:
    """Verifica que extra='forbid' rechaza campos no contratados."""
    invalid = valid_features_dict.copy()
    invalid["uncontracted_extra_field"] = "malicious_or_buggy_data"
    with pytest.raises(ValidationError):
        TabularFeatures(**invalid)


def test_business_invariant_charges(valid_features_dict: dict) -> None:
    """Verifica que total_charges no puede ser menor a monthly_charges si tenure >= 1."""
    invalid = valid_features_dict.copy()
    invalid["customer_tenure_months"] = 6
    invalid["monthly_charges"] = 100.0
    invalid["total_charges"] = 50.0  # Menor que monthly_charges
    with pytest.raises(ValidationError) as exc:
        TabularFeatures(**invalid)
    assert "total_charges no puede ser menor que monthly_charges" in str(exc.value)


def test_reject_nil_uuid(valid_tabular_features: TabularFeatures) -> None:
    """Verifica que el UUID nulo es rechazado."""
    with pytest.raises(ValidationError):
        PredictionRequest(
            request_id="00000000-0000-0000-0000-000000000000",
            features=valid_tabular_features,
        )
