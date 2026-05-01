"""Fold-safe categorical encoders.

The only rule: encoders MUST be fit on the training slice of a fold and
applied to validation + test. Never fit on the full dataset — that leaks
test-row labels into training. See §3 of the executive report for the
five-stage leak audit that motivated this class.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


class FoldSafeTargetEncoder:
    """Smoothed mean target encoder, leakage-safe by construction.

    For each category c in column X[col], the encoded value is

        enc(c) = (sum_y_c + smoothing * global_mean) /
                 (count_c + smoothing)

    Unseen categories at transform time get `global_mean`.

    Usage:
        enc = FoldSafeTargetEncoder(cols=["sub_agency", "vehicle_make"])
        enc.fit(X_train, y_train)
        X_train_enc = enc.transform(X_train)
        X_val_enc   = enc.transform(X_val)
        X_test_enc  = enc.transform(X_test)

    The encoded columns are named `{col}_te` and the originals are dropped
    by `transform()`.

    Bayesian smoothing parameter `smoothing=20` follows Micci-Barreca 2001
    (ACM SIGKDD Explorations 3(1):27-32). The shrinkage prevents rare
    categories from overfitting to their handful of training samples.
    """

    def __init__(
        self,
        cols: Iterable[str],
        smoothing: float = 20.0,
        fillna: str = "<NA>",
    ):
        self.cols = list(cols)
        self.smoothing = float(smoothing)
        self.fillna = fillna
        self._maps: dict[str, pd.Series] = {}
        self._global: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FoldSafeTargetEncoder":
        y = pd.Series(y).astype(float).reset_index(drop=True)
        for col in self.cols:
            if col not in X.columns:
                continue
            s = X[col].astype("object").fillna(self.fillna).reset_index(drop=True)
            stats = y.groupby(s).agg(["sum", "count"])
            global_mean = float(y.mean())
            enc = (
                (stats["sum"] + self.smoothing * global_mean)
                / (stats["count"] + self.smoothing)
            )
            self._maps[col] = enc.astype(float)
            self._global[col] = global_mean
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.cols:
            if col not in out.columns:
                continue
            s = out[col].astype("object").fillna(self.fillna)
            enc_map = self._maps.get(col)
            if enc_map is None:
                out[f"{col}_te"] = self._global.get(col, 0.0)
            else:
                out[f"{col}_te"] = (
                    s.map(enc_map).astype(float).fillna(self._global[col])
                )
            out = out.drop(columns=[col])
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def encoded_cols(self) -> List[str]:
        return [f"{c}_te" for c in self.cols]
