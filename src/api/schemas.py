"""Request validation schemas for the inference API."""

from marshmallow import Schema, fields, ValidationError, validate


class FeatureVectorSchema(Schema):
    """Schema for validating feature vectors in prediction requests."""

    Amount = fields.Float(required=True, validate=validate.Range(min=0))
    log_amount = fields.Float(required=True)

    hour_sin = fields.Float(required=True, validate=validate.Range(min=-1, max=1))
    hour_cos = fields.Float(required=True, validate=validate.Range(min=-1, max=1))

    dow_sin = fields.Float(required=True, validate=validate.Range(min=-1, max=1))
    dow_cos = fields.Float(required=True, validate=validate.Range(min=-1, max=1))

    month_sin = fields.Float(required=True, validate=validate.Range(min=-1, max=1))
    month_cos = fields.Float(required=True, validate=validate.Range(min=-1, max=1))

    is_weekend = fields.Int(required=True, validate=validate.OneOf([0, 1]))
    is_night = fields.Int(required=True, validate=validate.OneOf([0, 1]))

    currency_mismatch = fields.Int(required=True, validate=validate.OneOf([0, 1]))
    cross_border = fields.Int(required=True, validate=validate.OneOf([0, 1]))
    is_round_amount = fields.Int(required=True, validate=validate.OneOf([0, 1]))

    sender_txn_count_24h = fields.Int(required=True, validate=validate.Range(min=0))
    sender_amount_sum_24h = fields.Float(required=True, validate=validate.Range(min=0))
    sender_amount_mean_30d = fields.Float(required=True, allow_none=True)
    sender_amount_std_30d = fields.Float(required=True, allow_none=True)
    sender_amount_zscore = fields.Float(required=True)

    receiver_txn_count_24h = fields.Int(required=True, validate=validate.Range(min=0))
    receiver_amount_sum_24h = fields.Float(required=True, validate=validate.Range(min=0))

    seconds_since_sender_txn = fields.Int(required=True)

    sender_txn_count_lifetime = fields.Int(required=True, validate=validate.Range(min=0))
    receiver_txn_count_lifetime = fields.Int(required=True, validate=validate.Range(min=0))
    sender_out_degree = fields.Int(required=True, validate=validate.Range(min=0))
    receiver_in_degree = fields.Int(required=True, validate=validate.Range(min=0))
    pair_transaction_count = fields.Int(required=True, validate=validate.Range(min=0))
    sender_counterparty_hhi = fields.Float(required=True, validate=validate.Range(min=0, max=1))

    Payment_type = fields.Str(required=True)
    Payment_currency = fields.Str(required=True)
    Received_currency = fields.Str(required=True)
    Sender_bank_location = fields.Str(required=True)
    Receiver_bank_location = fields.Str(required=True)

    class Meta:
        unknown = "raise"


def validate_feature_vector(payload):
    """
    Validate a feature vector against the schema.

    Args:
        payload: Dictionary with feature values

    Returns:
        Tuple of (cleaned_data, errors)

    Raises:
        ValidationError: If validation fails
    """
    schema = FeatureVectorSchema()
    return schema.load(payload)
