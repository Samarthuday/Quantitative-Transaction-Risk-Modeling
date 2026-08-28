import shap


def create_shap_explainer(model):
    return shap.TreeExplainer(model)


def calculate_shap_values(
    model,
    X,
):
    explainer = create_shap_explainer(
        model
    )

    return explainer.shap_values(
        X
    )