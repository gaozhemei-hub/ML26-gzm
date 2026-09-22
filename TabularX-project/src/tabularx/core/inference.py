"""Frozen preprocessing state for inference on raw tables.

Stores fitted transformations, never training rows. Reusing this object must
not fit imputers/encoders on new prediction data.
"""
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class FittedPreprocessor(TransformerMixin, BaseEstimator):
    columns: dict[str, dict[str, Any]]
    output_columns: list[str]

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        raise RuntimeError("Frozen preprocessing state: create a new training run to refit")

    def transform(self, X):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("Prediction input must be a DataFrame with unique column names")
        required = [name for name, state in self.columns.items() if state["kind"] != "drop"]
        missing = set(required) - set(X.columns)
        if missing:
            raise ValueError(f"Missing raw input columns: {sorted(missing)}")
        parts = []
        for name, state in self.columns.items():
            kind = state["kind"]
            if kind == "drop":
                continue
            col = X[name]
            if kind == "datetime":
                d = pd.to_datetime(col)
                names = state["date_parts"]
                values = pd.DataFrame({f"{name}_{part}": getattr(d.dt, part) for part in names}, index=X.index)
            elif kind == "boolean":
                values = col.fillna(False).astype(int).to_frame()
            else:
                if not state["steps"]:
                    values = col.to_frame()
                else:
                    arr = col.to_numpy().reshape(-1, 1)
                    for operation, fitted in state["steps"]:
                        if operation == "frequency":
                            arr = pd.Series(arr.ravel()).map(fitted).fillna(0.0).to_numpy().reshape(-1, 1)
                        else:
                            arr = fitted.transform(arr)
                    values = pd.DataFrame(arr, index=X.index, columns=state["outputs"])
            parts.append(values)
        out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)
        if not out.columns.is_unique:
            raise ValueError("Transformed feature names are not unique")
        return out.loc[:, self.output_columns]

    def get_feature_names_out(self, input_features=None):
        import numpy as np
        return np.asarray(self.output_columns, dtype=object)
