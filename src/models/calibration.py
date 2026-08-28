import numpy as np
from sklearn.linear_model import LogisticRegression

EPSILON = 1e-6


def probability_to_logit(probability):
    probability = np.clip(
        probability,
        EPSILON,
        1 - EPSILON,
    )

    return np.log(
        probability / (1 - probability)
    )


class ProbabilityCalibrator:

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, probabilities, targets):
        targets = np.asarray(targets)
        if np.unique(targets).size < 2:
            raise ValueError("Probability calibration requires both target classes.")

        logits = probability_to_logit(
            np.asarray(probabilities)
        ).reshape(-1, 1)

        self.model.fit(
            logits,
            targets,
        )

        return self

    def predict(self, probabilities):
        logits = probability_to_logit(
            np.asarray(probabilities)
        ).reshape(-1, 1)

        return self.model.predict_proba(
            logits
        )[:, 1]
