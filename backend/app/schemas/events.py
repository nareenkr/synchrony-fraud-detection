"""Canonical input events for online and offline fraud decisioning."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from ipaddress import ip_address
from math import isfinite
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@-]*$",
    ),
]
Region = Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=64)]


class LendingChannel(StrEnum):
    WEB = "WEB"
    MOBILE = "MOBILE"
    PARTNER_API = "PARTNER_API"
    AGENT = "AGENT"


class LoanApplicationEvent(BaseModel):
    """A validated lending event before point-in-time feature computation.

    Optional observations remain optional: new applicants and partially observed
    channels must still be scoreable. Derived velocity and graph values are not
    accepted here; the shared feature builder computes those from prior state.
    """

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    application_id: Identifier
    user_id: Identifier
    event_timestamp: datetime
    requested_loan_amount: float = Field(gt=0, le=10_000_000)
    channel: LendingChannel = LendingChannel.WEB

    income: float | None = Field(default=None, gt=0, le=100_000_000)
    debt_to_income_ratio: float | None = Field(default=None, ge=0, le=10)
    account_age_days: int | None = Field(default=None, ge=0, le=36_500)
    bank_account_age_days: int | None = Field(default=None, ge=0, le=36_500)

    device_id: Identifier | None = None
    ip_address: str | None = Field(default=None, min_length=3, max_length=45)
    bank_account_id: Identifier | None = None
    geographic_region: Region | None = None

    device_changes_30d: int | None = Field(default=None, ge=0, le=1_000)
    login_frequency_24h: int | None = Field(default=None, ge=0, le=100_000)
    failed_login_attempts_24h: int | None = Field(default=None, ge=0, le=100_000)
    previous_rejected_applications: int | None = Field(default=None, ge=0, le=100_000)
    unusual_login_location: bool | None = None

    transaction_amount: float | None = Field(default=None, ge=0, le=100_000_000)
    transaction_frequency_24h: int | None = Field(default=None, ge=0, le=1_000_000)
    transaction_amount_deviation: float | None = Field(default=None, ge=0, le=1_000)
    origin_balance_before: float | None = Field(default=None, ge=0, le=1_000_000_000)
    origin_balance_after: float | None = Field(default=None, ge=0, le=1_000_000_000)

    @field_validator("event_timestamp")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_timestamp must include a UTC offset")
        return value

    @field_validator("ip_address")
    @classmethod
    def validate_ip_address(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return str(ip_address(value))
        except ValueError as exc:
            raise ValueError("ip_address must be a valid IPv4 or IPv6 address") from exc

    @field_validator(
        "requested_loan_amount",
        "income",
        "debt_to_income_ratio",
        "transaction_amount",
        "transaction_amount_deviation",
        "origin_balance_before",
        "origin_balance_after",
    )
    @classmethod
    def numbers_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("numeric inputs must be finite")
        return value

    @model_validator(mode="after")
    def balances_are_plausible(self) -> LoanApplicationEvent:
        # A negative post-transaction balance is rejected by the field bound;
        # balances are otherwise descriptive and need not reconcile for every
        # transaction type in PaySim.
        return self
