"""Tests for configuration management."""


from src.config import (
    PreprocessingConfig,
    TrainingConfig,
    XGBoostConfig,
    get_config,
    get_fast_config,
    get_production_config,
)


def test_xgboost_config_defaults():
    """Test XGBoost config has sensible defaults."""
    config = XGBoostConfig()

    assert config.n_estimators == 500
    assert config.max_depth == 5
    assert config.learning_rate == 0.05
    assert config.random_state == 42


def test_xgboost_config_to_dict():
    """Test XGBoost config can be converted to dict."""
    config = XGBoostConfig(n_estimators=100)

    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert config_dict["n_estimators"] == 100
    assert config_dict["learning_rate"] == 0.05


def test_preprocessing_config_defaults():
    """Test preprocessing config has sensible defaults."""
    config = PreprocessingConfig()

    assert config.numeric_impute_strategy == "median"
    assert config.categorical_impute_strategy == "most_frequent"
    assert config.categorical_min_frequency == 20


def test_training_config_auto_initialization():
    """Test that TrainingConfig initializes sub-configs automatically."""
    config = TrainingConfig()

    assert isinstance(config.xgboost, XGBoostConfig)
    assert isinstance(config.preprocessing, PreprocessingConfig)
    assert config.xgboost.n_estimators == 500


def test_fast_config():
    """Test fast/smoke test configuration."""
    config = get_fast_config()

    assert config.xgboost.n_estimators == 10
    assert isinstance(config, TrainingConfig)


def test_production_config():
    """Test production configuration."""
    config = get_production_config()

    assert config.xgboost.n_estimators == 500


def test_get_config_with_fast_mode():
    """Test get_config function with fast mode."""
    fast_config = get_config(fast=True)
    prod_config = get_config(fast=False)

    assert fast_config.xgboost.n_estimators == 10
    assert prod_config.xgboost.n_estimators == 500


def test_alert_config_initialization():
    """Test AlertConfig post_init."""
    from src.config import AlertConfig

    config = AlertConfig()

    assert config.alert_rate == 0.005
    assert isinstance(config.alert_rates_for_metrics, list)
    assert len(config.alert_rates_for_metrics) == 3
    assert 0.001 in config.alert_rates_for_metrics


def test_alert_config_custom_rates():
    """Test AlertConfig with custom rates."""
    from src.config import AlertConfig

    custom_rates = [0.01, 0.02, 0.05]
    config = AlertConfig(alert_rates_for_metrics=custom_rates)

    assert config.alert_rates_for_metrics == custom_rates


def test_config_customization():
    """Test that configs can be customized."""
    config = TrainingConfig()
    config.xgboost.n_estimators = 200
    config.xgboost.learning_rate = 0.1

    assert config.xgboost.n_estimators == 200
    assert config.xgboost.learning_rate == 0.1


def test_config_isolation():
    """Test that configs are isolated (no shared state)."""
    config1 = get_config(fast=True)
    config2 = get_config(fast=True)

    config1.xgboost.n_estimators = 50

    assert config2.xgboost.n_estimators == 10
