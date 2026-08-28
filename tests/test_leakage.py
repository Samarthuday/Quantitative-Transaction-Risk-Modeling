from src.models.train import MODEL_FEATURES


FORBIDDEN_FEATURES = {
    "Is_laundering",
    "Laundering_type",
    "Sender_account",
    "Receiver_account",
}


def test_no_identifier_or_target_leakage_features():
    assert not (set(MODEL_FEATURES) & FORBIDDEN_FEATURES)
