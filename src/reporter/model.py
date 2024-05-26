from typing import Protocol

import numpy as np
import pandas as pd


class ScikitModel(Protocol):
    def fit(self, x: pd.DataFrame, y: pd.Series, sample_weight=None) -> None: ...

    def predict(self, x: pd.DataFrame) -> np.ndarray: ...


class Model:
    def __init__(self, model: ScikitModel, x: pd.DataFrame, y: pd.Series, labels=None):
        self.model = model
        self.x = x
        self.y = y
        self.labels = np.unique(y) if labels is None else labels
        self.y_pred = model.predict(x)
        self.y_proba = None
        self.name = model.__class__.__name__

        if hasattr(model, 'pred_proba'):
            self.y_proba = model.pred_proba(x)
