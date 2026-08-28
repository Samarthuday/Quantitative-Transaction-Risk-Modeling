from sklearn.linear_model import SGDClassifier


def build_logistic_baseline():

    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        class_weight="balanced",
        max_iter=1000,
        tol=1e-3,
        random_state=42,
    )