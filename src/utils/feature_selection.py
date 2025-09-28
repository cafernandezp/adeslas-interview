# src/utils/feature_selection.py
from __future__ import annotations

from typing import Iterable, Tuple, List
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

__all__ = ["calc_iv", "calc_ks", "filter_vars", "top_xgb_vars"]

_EPS = 1e-6


def calc_iv(df: pd.DataFrame, feature: str, target: str, bins: int = 10) -> float:
    """Compute Information Value (IV) for a numeric feature using quantile bins.

    Args:
        df: DataFrame with at least `feature` and `target`.
        feature: Numeric column to evaluate (will dropna).
        target: Binary target column in {0,1}.
        bins: Number of quantile bins (duplicates="drop" aplicado).

    Returns:
        IV value (float). Higher -> more separation power.

    Raises:
        KeyError: If columns are missing.
        ValueError: If target is not binary after filtering.
    """
    s = df[[feature, target]].dropna()
    if s.empty:
        return 0.0

    # Validación binaria
    tvals = s[target].unique()
    if not set(pd.Series(tvals).dropna().unique()).issubset({0, 1}):
        raise ValueError(f"`target` must be binary in {{0,1}}. Found: {tvals}")

    # Bins por cuantiles (rank para evitar empates)
    s = s.assign(bin=pd.qcut(s[feature].rank(method="first"),
                             q=bins, duplicates="drop"))

    tab = s.groupby("bin", observed=True)[target].agg(["count", "sum"])
    tab = tab.rename(columns={"count": "n", "sum": "bad"})
    tab["good"] = tab["n"] - tab["bad"]

    dist_good = tab["good"] / max(tab["good"].sum(), _EPS)
    dist_bad = tab["bad"] / max(tab["bad"].sum(), _EPS)

    woe = np.log((dist_good + _EPS) / (dist_bad + _EPS))
    iv = ((dist_good - dist_bad) * woe).sum()
    return float(iv)


def calc_ks(df: pd.DataFrame, feature: str, target: str) -> float:
    """Compute KS statistic between positive and negative classes for a numeric feature.

    Args:
        df: DataFrame with `feature` and `target`.
        feature: Numeric column to evaluate (will dropna).
        target: Binary target column in {0,1}.

    Returns:
        KS statistic in [0,1].

    Raises:
        KeyError: If columns are missing.
        ValueError: If target is not binary after filtering.
    """
    s = df[[feature, target]].dropna().sort_values(feature)
    if s.empty:
        return 0.0

    tvals = s[target].unique()
    if not set(pd.Series(tvals).dropna().unique()).issubset({0, 1}):
        raise ValueError(f"`target` must be binary in {{0,1}}. Found: {tvals}")

    # CDFs acumuladas
    s = s.assign(cum_good=(1 - s[target]).cumsum(),
                 cum_bad=s[target].cumsum())
    cum_good = s["cum_good"] / max(s["cum_good"].iloc[-1], _EPS)
    cum_bad = s["cum_bad"] / max(s["cum_bad"].iloc[-1], _EPS)
    ks = float(np.max(np.abs(cum_good - cum_bad)))
    return ks


def filter_vars(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    iv_min: float = 0.02,
    ks_min: float = 0.05,
    bins: int = 10,
) -> Tuple[List[str], pd.DataFrame]:
    """Filter features by IV and KS thresholds.

    Args:
        df: DataFrame con features y target.
        features: Lista de nombres de columnas a evaluar (numéricas).
        target: Nombre del target binario {0,1}.
        iv_min: Umbral mínimo de IV para pasar el filtro.
        ks_min: Umbral mínimo de KS para pasar el filtro.
        bins: Número de cuantiles para IV.

    Returns:
        selected: Lista de features que pasan ambos umbrales.
        df_metrics: DataFrame con columnas ["feature","iv","ks"].
    """
    iv_dict, ks_dict = {}, {}
    for f in features:
        try:
            iv_dict[f] = calc_iv(df, f, target, bins=bins)
            ks_dict[f] = calc_ks(df, f, target)
        except Exception:
            # Ignora features problemáticas (no numéricas, constantes, etc.)
            continue

    if not iv_dict:
        return [], pd.DataFrame(columns=["feature", "iv", "ks"])

    feats = list(iv_dict.keys())
    df_metrics = pd.DataFrame({
        "feature": feats,
        "iv": [iv_dict[f] for f in feats],
        "ks": [ks_dict.get(f, np.nan) for f in feats],
    })

    df_metrics = df_metrics.sort_values(["iv", "ks"], ascending=False, na_position="last")
    selected = df_metrics.query("iv >= @iv_min and ks >= @ks_min")["feature"].tolist()
    return selected, df_metrics


def top_xgb_vars(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    top_n: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit a quick XGBoost and return top-N features by gain-based importance.

    Args:
        df: DataFrame con features y target.
        features: Lista de columnas (se asume numéricas).
        target: Nombre del target binario {0,1}.
        top_n: Número de variables a devolver.
        random_state: Semilla de reproducibilidad.

    Returns:
        DataFrame con columnas ["feature","importance"] ordenado descendente.
    """
    if len(features) == 0:
        return pd.DataFrame(columns=["feature", "importance"])

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        eta=0.05,
        gamma=5,
        subsample=0.6,
        colsample_bytree=1.0,
        max_depth=3,
        nthread=8,
        min_child_weight=30,
        n_estimators=300,
        random_state=random_state,
        verbosity=0,
    )
    model.fit(df[features], df[target].astype(int))
    importances = getattr(model, "feature_importances_", np.zeros(len(features)))

    df_imp = pd.DataFrame({"feature": features, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False, kind="mergesort")
    return df_imp.head(top_n).reset_index(drop=True)
