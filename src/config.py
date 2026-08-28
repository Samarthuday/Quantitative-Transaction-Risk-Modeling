"""Configuration management for model training and experiments."""

from dataclasses import dataclass, asdict


@dataclass
class XGBoostConfig:
    """XGBoost hyperparameter configuration."""

    n_estimators: int = 500
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    reg_alpha: float = 0.1
    reg_lambda: float = 2.0
    tree_method: str = "hist"
    eval_metric: str = "aucpr"
    random_state: int = 42

    def to_dict(self):
        """Convert to dictionary for XGBoost initialization."""
        return asdict(self)


@dataclass
class PreprocessingConfig:
    """Data preprocessing configuration."""

    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"
    categorical_min_frequency: int = 20
    categorical_handle_unknown: str = "ignore"


@dataclass
class TemporalSplitConfig:
    """Temporal split configuration."""

    train_fraction: float = 0.65
    calibration_fraction: float = 0.10
    validation_fraction: float = 0.10


@dataclass
class AlertConfig:
    """Alert threshold and budget configuration."""

    alert_rate: float = 0.005  # 0.5%
    alert_rates_for_metrics: list = None

    def __post_init__(self):
        if self.alert_rates_for_metrics is None:
            self.alert_rates_for_metrics = [0.001, 0.005, 0.01]


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    xgboost: XGBoostConfig = None
    preprocessing: PreprocessingConfig = None
    temporal_split: TemporalSplitConfig = None
    alert: AlertConfig = None

    def __post_init__(self):
        if self.xgboost is None:
            self.xgboost = XGBoostConfig()
        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig()
        if self.temporal_split is None:
            self.temporal_split = TemporalSplitConfig()
        if self.alert is None:
            self.alert = AlertConfig()


def get_fast_config() -> TrainingConfig:
    """Return configuration for fast/smoke testing."""
    config = TrainingConfig()
    config.xgboost.n_estimators = 10  # Minimal trees for fast testing
    return config


def get_production_config() -> TrainingConfig:
    """Return configuration for production training."""
    return TrainingConfig()


def get_config(fast: bool = False) -> TrainingConfig:
    """Get configuration based on mode."""
    return get_fast_config() if fast else get_production_config()
