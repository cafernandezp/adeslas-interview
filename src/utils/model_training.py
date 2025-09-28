from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping  # <-- añade este import arriba



# =========================
# Métricas auxiliares
# =========================
def to_1d_series(y: pd.DataFrame | pd.Series) -> pd.Series:
    """Return a 1D numeric Series from y (DataFrame or Series).

    Args:
        y: Target as a Series or single-column DataFrame.

    Returns:
        A numeric pandas Series with shape (n,).
    """
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"y has {y.shape[1]} columns; expected 1.")
        y = y.iloc[:, 0]
    return pd.to_numeric(y.squeeze(), errors="coerce")


def ks_score(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    """Compute KS statistic for binary classification.

    KS = max_s |F1(s) - F0(s)| over the score domain.

    Args:
        y_true: Iterable of binary labels {0,1}.
        y_score: Iterable of predicted probabilities/scores in [0,1].

    Returns:
        KS statistic as a float.
    """
    df = pd.DataFrame({"y": y_true, "p": y_score}).dropna()
    if df["y"].sum() == 0 or (1 - df["y"]).sum() == 0:
        return np.nan  # KS no definido si no hay ambas clases
    df = df.sort_values("p")
    cdf1 = (df["y"].cumsum()) / df["y"].sum()              # positivos
    cdf0 = ((1 - df["y"]).cumsum()) / (1 - df["y"]).sum()  # negativos
    return float(np.nanmax(np.abs(cdf1 - cdf0)))


# =========================
# Config y resultados
# =========================
@dataclass(slots=True)
class TrainResults:
    """Container for training artifacts/results."""
    model: XGBClassifier
    used_features: List[str]
    metrics: pd.DataFrame  # index: split; cols: auc, ks, prauc, n
    best_iteration: int | None = None
    best_n_estimators: int | None = None


def default_xgb_params() -> Dict[str, object]:
    """Return a robust default config for binary XGBoost (CPU).

    Returns:
        Dict of XGBClassifier keyword arguments.
    """
    return dict(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
    )


# =========================
# Entrenamiento y evaluación
# =========================
def train_xgb_binary(
    X_train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series | pd.DataFrame,
    selected_vars: Sequence[str],
    xgb_params: Mapping[str, object] | None = None,
    early_stopping_rounds: int = 100,
) -> TrainResults:
    """Train XGBoost binary using xgboost.train for early stopping (XGB 3.x),
    then re-fit an XGBClassifier with best_iteration on train only.

    Steps:
      1) Select and order features.
      2) Use xgb.train with DMatrix (train/val) to get best_iteration by AUC.
      3) Refit sklearn XGBClassifier with n_estimators = best_iteration+1 on train.
      4) Return model, used_features, and train/val metrics.
    """
    import xgboost as xgb

    params = default_xgb_params() if xgb_params is None else dict(xgb_params)

    # Asegura Series 1D
    y_tr = to_1d_series(y_train)
    y_va = to_1d_series(y_val)

    # Filtra columnas (intersección y orden)
    used_features = [c for c in selected_vars if c in X_train.columns]
    if not used_features:
        raise ValueError("No hay features en la intersección de selected_vars y X_train.columns.")

    X_tr = X_train[used_features].copy()
    X_va = X_val[used_features].copy()

    # --- 1) Early stopping con xgb.train ---
    # Mapear parámetros al formato del booster
    booster_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": int(params.get("max_depth", 6)),
        "subsample": float(params.get("subsample", 0.8)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
        "alpha": float(params.get("reg_alpha", 0.0)),      # reg_alpha
        "lambda": float(params.get("reg_lambda", 1.0)),    # reg_lambda
        "tree_method": params.get("tree_method", "hist"),
        "eta": float(params.get("learning_rate", 0.05)),   # learning_rate -> eta
        "seed": int(params.get("random_state", 42)),
    }

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    watchlist = [(dtrain, "train"), (dval, "val")]

    num_boost_round = int(params.get("n_estimators", 2000))
    booster = xgb.train(
        booster_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    best_it = booster.best_iteration
    if best_it is None:
        best_it = booster.best_ntree_limit - 1
    best_ne = int(best_it + 1)

    # Refit sklearn wrapper con árboles efectivos
    clf_params = dict(params)
    clf_params["n_estimators"] = best_ne
    model = XGBClassifier(**clf_params)
    model.fit(X_tr, y_tr, verbose=False)

    # --- 3) Métricas train/val ---
    metrics = evaluate_model(
        model,
        splits={"train": (X_tr, y_tr), "val": (X_va, y_va)},
    )

    return TrainResults(
        model=model,
        used_features=used_features,
        metrics=metrics,
        best_iteration=int(best_it),
        best_n_estimators=best_ne,
    )



def evaluate_model(
    model: XGBClassifier,
    splits: Mapping[str, Tuple[pd.DataFrame, pd.Series | pd.DataFrame]],
) -> pd.DataFrame:
    """Evaluate model on one or more splits (AUC, KS, PR-AUC, N).

    Args:
        model: Trained XGBClassifier.
        splits: Dict split_name -> (X, y).

    Returns:
        DataFrame indexed by split with columns ['auc','ks','prauc','n'].
    """
    rows: List[Dict[str, float | int | str]] = []
    for name, (X, y) in splits.items():
        y1d = to_1d_series(y)
        proba = model.predict_proba(X)[:, 1]
        rows.append(
            dict(
                split=name,
                auc=roc_auc_score(y1d, proba),
                ks=ks_score(y1d, proba),
                prauc=average_precision_score(y1d, proba),
                n=int(len(y1d)),
            )
        )
    return pd.DataFrame(rows).set_index("split").sort_index()


# =========================
# Persistencia
# =========================
def save_artifacts(
    results: TrainResults,
    out_dir: Path,
    model_name: str = "xgb_model_selected_vars.joblib",
    features_name: str = "xgb_used_features.csv",
) -> Tuple[Path, Path]:
    """Persist model and used feature list to disk.

    Args:
        results: TrainResults returned by `train_xgb_binary`.
        out_dir: Directory where artifacts will be stored.
        model_name: File name for the serialized model.
        features_name: File name for the features CSV.

    Returns:
        A tuple with (model_path, features_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / model_name
    feats_path = out_dir / features_name

    joblib.dump(results.model, model_path)
    pd.Series(results.used_features, name="feature").to_csv(feats_path, index=False)
    return model_path, feats_path
