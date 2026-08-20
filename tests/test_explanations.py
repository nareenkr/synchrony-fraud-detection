from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.explainability import (
    ExplanationService,
    FeatureContribution,
    ModelContributionAdapter,
    PerturbationContributionAdapter,
    ShapContributionAdapter,
)
from backend.app.schemas import RiskSignal


def test_curated_reasons_are_human_readable_ranked_and_deduplicated() -> None:
    service = ExplanationService(max_reasons=3)
    result = service.explain(
        features={
            "account_age_days": 2,
            "applications_last_hour": 7,
            "shared_device_user_count": 5,
        },
        contributions=[
            FeatureContribution("account_age_days", 0.5, 2),
            FeatureContribution("applications_last_hour", 0.9, 7),
            FeatureContribution("shared_device_user_count", 0.7, 5),
        ],
        component_signals=[
            RiskSignal(
                code="RAPID_APPLICATIONS",
                message="untrusted duplicate text",
                severity=0.95,
                source="behavioral",
            ),
            RiskSignal(
                code="DEVICE_SHARED",
                message="untrusted duplicate text",
                severity=0.8,
                source="graph",
            ),
        ],
    )

    assert [signal.code for signal in result.signals] == [
        "APPLICATION_VELOCITY",
        "SHARED_DEVICE",
        "YOUNG_ACCOUNT",
    ]
    assert result.reasons[0] == "Several loan applications were submitted within one hour"
    assert len(result.reasons) == len(set(result.reasons))
    assert all("_" not in reason for reason in result.reasons)


def test_output_never_exposes_raw_shap_values_identifiers_or_untrusted_messages() -> None:
    result = ExplanationService().explain(
        features={"shared_ip_user_count": 9},
        contributions=[FeatureContribution("shared_ip_user_count", 0.823746, 9)],
        component_signals=[
            {
                "code": "IP_192_168_1_22",
                "message": "user USER-77 at device DEV-secret has SHAP 0.823746",
                "severity": 0.9,
                "source": "graph",
            }
        ],
    )
    serialized = (
        " ".join(result.reasons)
        + " "
        + " ".join(signal.model_dump_json() for signal in result.signals)
    )
    for secret in ("192", "USER-77", "DEV-secret", "SHAP", "0.823746"):
        assert secret not in serialized
    assert "Network is associated with multiple accounts" in result.reasons
    assert "Account relationships show an unusual sharing pattern" in result.reasons


def test_missing_optional_context_and_unavailable_model_explanation_are_safe() -> None:
    result = ExplanationService().explain(
        features=None,
        model=None,
        contributions=None,
        component_signals=[
            {"code": "ANOMALY_OUTLIER", "severity": 0.75, "source": "anomaly"},
            {"code": "BROKEN", "severity": "not-a-number", "source": "behavioral"},
            {"code": "IGNORED", "severity": 1.0, "source": "unknown"},
        ],
    )
    assert result.reasons == ["Activity differs substantially from established patterns"]
    assert result.signals[0].code == "ANOMALOUS_ACTIVITY"


def test_example_risk_reasons_cover_lending_behavior_and_fraud_ring_context() -> None:
    result = ExplanationService(max_reasons=10).explain(
        features={
            "loan_to_income_ratio": 1.4,
            "account_age_days": 2,
            "is_new_device": 1,
            "applications_last_hour": 7,
            "shared_device_user_count": 5,
            "shared_ip_user_count": 8,
        },
        contributions=[
            FeatureContribution("loan_to_income_ratio", 0.8, 1.4),
            FeatureContribution("account_age_days", 0.7, 2),
            FeatureContribution("is_new_device", 0.4, 1),
        ],
        component_signals=[
            {"code": "RAPID_APPLICATIONS", "severity": 0.9, "source": "behavioral"},
            {"code": "DEVICE_SHARED", "severity": 0.85, "source": "graph"},
            {"code": "IP_SHARED", "severity": 0.8, "source": "graph"},
        ],
    )
    assert {
        "Requested loan is high relative to declared income",
        "Account was created recently",
        "Application was submitted from a new device",
        "Several loan applications were submitted within one hour",
        "Device is associated with multiple accounts",
        "Network is associated with multiple accounts",
    }.issubset(result.reasons)


def test_model_agnostic_adapter_and_pipeline_integration_produce_local_contributions(
    monkeypatch,
) -> None:
    columns = ["loan_to_income_ratio", "account_age_days"]
    training = pd.DataFrame([[0.1, 900], [0.2, 600], [1.2, 3], [1.6, 1]], columns=columns)
    labels = np.array([0, 0, 1, 1])
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=7)).fit(
        training, labels
    )
    row = pd.DataFrame([[1.5, 2]], columns=columns)

    fallback = PerturbationContributionAdapter(
        {"loan_to_income_ratio": 0.2, "account_age_days": 600}
    ).explain(model, row)
    assert {item.feature for item in fallback} == set(columns)
    assert all(np.isfinite(item.contribution) for item in fallback)

    shap_calls: list[str] = []

    class FakeLinearExplainer:
        def __init__(self, estimator, background) -> None:
            assert hasattr(estimator, "coef_")
            assert np.asarray(background).shape == (1, 2)
            shap_calls.append("linear")

        def __call__(self, matrix):
            assert np.asarray(matrix).shape == (1, 2)
            return SimpleNamespace(values=np.array([[0.8, 0.6]]))

    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(LinearExplainer=FakeLinearExplainer))
    integrated = ModelContributionAdapter().explain(model, row)
    direct = ShapContributionAdapter().explain(model, row)
    # Runtime linear explanations use the fast perturbation equivalent; the
    # explicit SHAP adapter remains available for retained offline evidence.
    assert shap_calls == ["linear"]
    assert [item.feature for item in integrated] == columns
    assert [item.value for item in direct] == [1.5, 2.0]

    explanation = ExplanationService(
        adapter=PerturbationContributionAdapter(
            {"loan_to_income_ratio": 0.2, "account_age_days": 600}
        )
    ).explain(features=row, model=model)
    assert any(
        "loan" in reason.lower() or "account" in reason.lower() for reason in explanation.reasons
    )
