"""Calibrated current-score and recurrent temporal forecasting models."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from app.ml.features import FEATURE_NAMES
from app.ml.preprocessor import DEFAULT_SCALER_PATH, FeaturePreprocessor, clip_health

MODEL_PATH = Path(__file__).resolve().parent / "training" / "phps_xgboost_fusion.json"
SEQUENCE_MODEL_PATH = Path(__file__).resolve().parent / "training" / "phps_gru_forecaster.pt"


class CalibratedXGBoost:
    """XGBoost inference with the exact persisted training calibration."""
    def __init__(self, model_path: Path = MODEL_PATH, scaler_path: Path = DEFAULT_SCALER_PATH):
        self.model_path, self.preprocessor = model_path, FeaturePreprocessor.load(scaler_path)
        self.model = None

    def _load(self):
        if self.model is None:
            import xgboost as xgb
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Current-health model missing: {self.model_path}")
            self.model = xgb.XGBRegressor()
            self.model.load_model(self.model_path)
        return self.model

    @property
    def calibrated(self) -> bool:
        return self.preprocessor._fitted

    def predict(self, features: dict) -> float:
        matrix = self.preprocessor.transform_matrix([features])
        return clip_health(self._load().predict(matrix)[0])


def build_gru(input_size: int = len(FEATURE_NAMES), hidden_size: int = 32, horizon: int = 3):
    """Small GRU that maps four weekly snapshots to three future health scores."""
    import torch
    from torch import nn

    class HealthGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, horizon))

        def forward(self, sequence):
            _, hidden = self.gru(sequence)
            return self.head(hidden[-1])
    return HealthGRU()


class TemporalHealthForecaster:
    """Loads a trained GRU; otherwise uses a transparent history-only baseline."""
    def __init__(self, path: Path = SEQUENCE_MODEL_PATH, horizon: int = 3):
        self.path, self.horizon, self._model = path, horizon, None

    def _load(self):
        if not self.path.is_file():
            return None
        if self._model is None:
            import torch
            self._model = build_gru(horizon=self.horizon)
            self._model.load_state_dict(torch.load(self.path, map_location="cpu", weights_only=True))
            self._model.eval()
        return self._model

    def forecast(self, sequence: np.ndarray, health_history: list[float]) -> dict:
        model = self._load()
        if model is not None:
            import torch
            with torch.no_grad():
                forecast = model(torch.tensor(sequence[-1:], dtype=torch.float32)).numpy()[0]
            return {"scores": [round(clip_health(x), 2) for x in forecast], "model_source": "trained_gru"}
        # Fallback is never a what-if override: it continues observed momentum.
        recent = health_history[-4:]
        slope = float(np.mean(np.diff(recent))) if len(recent) >= 2 else 0.0
        start = recent[-1] if recent else 50.0
        return {"scores": [round(clip_health(start + slope * week), 2) for week in range(1, self.horizon + 1)], "model_source": "history_trend_fallback"}
