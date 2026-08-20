"""Curated, identifier-free fraud reason catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ReasonTemplate:
    code: str
    feature: str
    risky_when: str
    threshold: float
    message: str

    def matches(self, value: float) -> bool:
        return value <= self.threshold if self.risky_when == "low" else value >= self.threshold


FEATURE_REASONS: Final[dict[str, ReasonTemplate]] = {
    "loan_to_income_ratio": ReasonTemplate(
        "HIGH_LOAN_TO_INCOME",
        "loan_to_income_ratio",
        "high",
        0.75,
        "Requested loan is high relative to declared income",
    ),
    "account_age_days": ReasonTemplate(
        "YOUNG_ACCOUNT",
        "account_age_days",
        "low",
        30.0,
        "Account was created recently",
    ),
    "bank_account_age_days": ReasonTemplate(
        "YOUNG_BANK_ACCOUNT",
        "bank_account_age_days",
        "low",
        30.0,
        "Linked bank account was established recently",
    ),
    "applications_last_hour": ReasonTemplate(
        "APPLICATION_VELOCITY",
        "applications_last_hour",
        "high",
        3.0,
        "Several loan applications were submitted within one hour",
    ),
    "applications_last_day": ReasonTemplate(
        "DAILY_APPLICATION_VELOCITY",
        "applications_last_day",
        "high",
        6.0,
        "Application frequency during the last day is unusually high",
    ),
    "transaction_amount_deviation": ReasonTemplate(
        "UNUSUAL_TRANSACTION_AMOUNT",
        "transaction_amount_deviation",
        "high",
        3.0,
        "Transaction amount is well above the applicant's recent pattern",
    ),
    "device_changes_30d": ReasonTemplate(
        "FREQUENT_DEVICE_CHANGES",
        "device_changes_30d",
        "high",
        3.0,
        "Device usage changed frequently during the last 30 days",
    ),
    "failed_login_attempts_24h": ReasonTemplate(
        "REPEATED_FAILED_LOGINS",
        "failed_login_attempts_24h",
        "high",
        3.0,
        "Several failed login attempts occurred during the last day",
    ),
    "previous_rejected_applications": ReasonTemplate(
        "PREVIOUS_REJECTIONS",
        "previous_rejected_applications",
        "high",
        1.0,
        "The applicant has previous rejected applications",
    ),
    "is_new_device": ReasonTemplate(
        "NEW_DEVICE",
        "is_new_device",
        "high",
        1.0,
        "Application was submitted from a new device",
    ),
    "unusual_login_location": ReasonTemplate(
        "UNUSUAL_LOCATION",
        "unusual_login_location",
        "high",
        1.0,
        "Login location differs from the applicant's established pattern",
    ),
    "shared_device_user_count": ReasonTemplate(
        "SHARED_DEVICE",
        "shared_device_user_count",
        "high",
        2.0,
        "Device is associated with multiple accounts",
    ),
    "shared_ip_user_count": ReasonTemplate(
        "SHARED_IP",
        "shared_ip_user_count",
        "high",
        3.0,
        "Network is associated with multiple accounts",
    ),
    "shared_bank_user_count": ReasonTemplate(
        "SHARED_BANK_ACCOUNT",
        "shared_bank_user_count",
        "high",
        2.0,
        "Linked bank account is associated with multiple applicants",
    ),
    "hour_of_day_deviation": ReasonTemplate(
        "UNUSUAL_APPLICATION_TIME",
        "hour_of_day_deviation",
        "high",
        6.0,
        "Application time differs substantially from the applicant's usual pattern",
    ),
    "is_night_application": ReasonTemplate(
        "NIGHT_APPLICATION",
        "is_night_application",
        "high",
        1.0,
        "Application was submitted during unusual overnight hours",
    ),
}


CODE_ALIASES: Final[dict[str, str]] = {
    "MANY_APPLICATIONS_LAST_HOUR": "APPLICATION_VELOCITY",
    "HIGH_APPLICATION_VELOCITY": "APPLICATION_VELOCITY",
    "RAPID_APPLICATIONS": "APPLICATION_VELOCITY",
    "DEVICE_SHARED": "SHARED_DEVICE",
    "MULTIPLE_USERS_DEVICE": "SHARED_DEVICE",
    "IP_SHARED": "SHARED_IP",
    "MULTIPLE_USERS_IP": "SHARED_IP",
    "BANK_SHARED": "SHARED_BANK_ACCOUNT",
    "LARGE_LOAN_REQUEST": "HIGH_LOAN_TO_INCOME",
    "HIGH_LOAN_AMOUNT": "HIGH_LOAN_TO_INCOME",
    "ACCOUNT_TOO_NEW": "YOUNG_ACCOUNT",
    "FAILED_LOGINS": "REPEATED_FAILED_LOGINS",
    "DEVICE_CHANGE": "NEW_DEVICE",
    "UNUSUAL_LOGIN_LOCATION": "UNUSUAL_LOCATION",
}


MESSAGES: Final[dict[str, str]] = {
    template.code: template.message for template in FEATURE_REASONS.values()
}
MESSAGES.update(
    {
        "ANOMALOUS_ACTIVITY": "Activity differs substantially from established patterns",
        "BEHAVIORAL_RISK": "Recent activity triggered behavioral risk controls",
        "GRAPH_LINKAGE": "Account relationships show an unusual sharing pattern",
        "ELEVATED_MODEL_RISK": "The fraud model identified an elevated combination of risk factors",
    }
)

SOURCE_FALLBACK: Final[dict[str, tuple[str, str]]] = {
    "supervised": ("ELEVATED_MODEL_RISK", MESSAGES["ELEVATED_MODEL_RISK"]),
    "anomaly": ("ANOMALOUS_ACTIVITY", MESSAGES["ANOMALOUS_ACTIVITY"]),
    "behavioral": ("BEHAVIORAL_RISK", MESSAGES["BEHAVIORAL_RISK"]),
    "graph": ("GRAPH_LINKAGE", MESSAGES["GRAPH_LINKAGE"]),
}


def canonical_feature_name(name: str) -> str:
    """Remove transformer prefixes while only accepting catalogued names."""

    candidate = name.rsplit("__", 1)[-1]
    if candidate in FEATURE_REASONS:
        return candidate
    return ""
