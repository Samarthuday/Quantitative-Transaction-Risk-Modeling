"""Pydantic models for API request/response validation."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class FeatureVector(BaseModel):
    """Transaction feature vector for risk prediction."""

    Amount: float = Field(..., ge=0, description="Transaction amount in base currency")
    log_amount: float = Field(..., description="Log(1 + Amount)")

    hour_sin: float = Field(..., ge=-1, le=1, description="Cyclical hour encoding")
    hour_cos: float = Field(..., ge=-1, le=1)
    dow_sin: float = Field(..., ge=-1, le=1, description="Day of week cyclical encoding")
    dow_cos: float = Field(..., ge=-1, le=1)
    month_sin: float = Field(..., ge=-1, le=1, description="Month cyclical encoding")
    month_cos: float = Field(..., ge=-1, le=1)

    is_weekend: int = Field(..., ge=0, le=1)
    is_night: int = Field(..., ge=0, le=1)
    currency_mismatch: int = Field(..., ge=0, le=1)
    cross_border: int = Field(..., ge=0, le=1)
    is_round_amount: int = Field(..., ge=0, le=1)

    sender_txn_count_24h: int = Field(..., ge=0)
    sender_amount_sum_24h: float = Field(..., ge=0)
    sender_amount_mean_30d: Optional[float] = Field(None)
    sender_amount_std_30d: Optional[float] = Field(None)
    sender_amount_zscore: float

    receiver_txn_count_24h: int = Field(..., ge=0)
    receiver_amount_sum_24h: float = Field(..., ge=0)

    seconds_since_sender_txn: int

    sender_txn_count_lifetime: int = Field(..., ge=0)
    receiver_txn_count_lifetime: int = Field(..., ge=0)
    sender_out_degree: int = Field(..., ge=0)
    receiver_in_degree: int = Field(..., ge=0)
    pair_transaction_count: int = Field(..., ge=0)
    sender_counterparty_hhi: float = Field(..., ge=0, le=1)

    Payment_type: str
    Payment_currency: str
    Received_currency: str
    Sender_bank_location: str
    Receiver_bank_location: str

    model_config = {"extra": "forbid"}


class HealthResponse(BaseModel):
    """Service health status."""

    status: str = Field(..., description="healthy or model_unavailable")
    model_loaded: bool
    timestamp: str
    detail: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Model metadata and evaluation metrics."""

    model: str
    model_version: str
    alert_rate: float
    decision_threshold: float
    test_metrics: Dict[str, Any]


class PredictionResponse(BaseModel):
    """Risk prediction result."""

    risk_probability: float = Field(..., ge=0, le=1)
    requires_review: bool
    risk_percentile: Optional[float] = Field(None, ge=0, le=100)
