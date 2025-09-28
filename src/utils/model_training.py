from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping 
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import precision_recall_curve



# -----------------------------------------------------------------------------
# randomized_search_xgb         -> HPO rápido con RandomizedSearchCV (CV estratificada).
# ThresholdReport (dataclass)   -> Contenedor de métricas a umbral (precision, recall, Fβ, etc.).
# threshold_metrics             -> Calcula precision/recall/Fβ en un umbral dado.
# select_threshold_by_fbeta     -> Selecciona umbral que maximiza Fβ (β>1 favorece recall).
# select_topk_by_budget         -> Umbral aproximado para seleccionar top-k según presupuesto (k = B/coste).
# shap_contribs                 -> Importancias SHAP nativas de XGBoost e interacciones (sin librería shap).
# incentive_plans               -> Escenarios por top-k% o thresholds: coste, precision, recall, Fβ, score medio.
# select_threshold_with_constraints -> Umbral que maximiza Fβ sujeto a coste ≤ presupuesto y precision mínima.
# campaign_scenarios            -> Planes por presupuesto con ROI esperado (usando supuestos de uplift y beneficio).
# =============================================================================




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




# =========================
# HPO con RandomizedSearchCV
# =========================
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

def randomized_search_xgb(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    base_params: Mapping[str, object] | None = None,
    n_iter: int = 40,
    cv_splits: int = 5,
    scoring: str = "roc_auc",  # "average_precision" si priorizas PR-AUC
    random_state: int = 42,
    param_distributions: Mapping[str, object] | None = None,  # opcional
) -> RandomizedSearchCV:
    """Run RandomizedSearchCV over XGBClassifier (sin early stopping, xgboost 3.x)."""
    y1d = to_1d_series(y)

    # 1) Merge de defaults + overrides del usuario
    merged = dict(default_xgb_params())
    if base_params:
        merged.update(base_params)

    # 2) Forzar solo una vez los campos operativos
    merged.update({
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "auc",
        "random_state": random_state,
        "n_estimators": 600,    # moderado; sin early stopping aquí
        "missing": np.nan,
    })
    base = XGBClassifier(**merged)

    # 3) Espacio de hiperparámetros (si no te pasan uno, usa el por defecto)
    if param_distributions is None:
        param_distributions = {
            "max_depth": randint(3, 7),
            "learning_rate": loguniform(0.01, 0.2),
            "subsample": uniform(0.5, 0.5),            # [0.5, 1.0]
            "colsample_bytree": uniform(0.5, 0.5),     # [0.5, 1.0]
            "min_child_weight": randint(1, 12),
            "gamma": uniform(0.0, 3.0),
            "reg_alpha": loguniform(1e-3, 10.0),
            "reg_lambda": loguniform(1e-2, 50.0),
        }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=-1,
        cv=cv,
        verbose=1,
        refit=True,
        random_state=random_state,
    )
    rs.fit(X, y1d)
    return rs



# =========================
# Métricas a umbral
# =========================
@dataclass(slots=True)
class ThresholdReport:
    """Container for threshold-based metrics."""

    threshold: float
    precision: float
    recall: float
    fbeta: float
    support_pos: int      # # de positivos reales
    positives_pred: int   # # de predichos positivos (contactados)

def threshold_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    beta: float = 2.0,
) -> ThresholdReport:
    """Compute precision/recall/Fβ at a probability threshold.

    Args:
      y_true: Etiquetas reales (0/1).
      y_proba: Probabilidades estimadas en [0,1].
      threshold: Umbral de decisión.
      beta: Ponderación de recall (β>1 favorece recall).

    Returns:
      ThresholdReport con métricas en el umbral dado.
    """
    y = np.asarray(y_true).astype(int)
    p = (y_proba >= threshold).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision == 0 and recall == 0:
        fbeta = 0.0
    else:
        b2 = beta**2
        denom = b2 * precision + recall
        fbeta = (1 + b2) * (precision * recall) / denom if denom > 0 else 0.0

    return ThresholdReport(
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        fbeta=float(fbeta),
        support_pos=int(y.sum()),
        positives_pred=int(p.sum()),
    )

def select_threshold_by_fbeta(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    beta: float = 2.0,
    grid: np.ndarray | None = None,
) -> ThresholdReport:
    """Select threshold that maximizes Fβ (β>1 favorece recall).

    Args:
      y_true: Etiquetas reales.
      y_proba: Probabilidades.
      beta: Peso del recall.
      grid: Vector de umbrales a evaluar (default: 0.05..0.95).

    Returns:
      ThresholdReport del mejor umbral por Fβ.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    reports = [threshold_metrics(y_true, y_proba, t, beta=beta) for t in grid]
    return max(reports, key=lambda r: r.fbeta)

def select_topk_by_budget(
    y_proba: np.ndarray,
    per_unit_cost: float,
    total_budget: float,
) -> float:
    """Return threshold that approximately selects top-k where k=floor(B/coste).

    Args:
      y_proba: Probabilidades estimadas.
      per_unit_cost: Coste por incentivo/cliente (p.ej., 50€).
      total_budget: Presupuesto total disponible.

    Returns:
      Umbral aproximado que deja ~k instancias por encima (top-k en score).
    """
    n = len(y_proba)
    max_units = int(total_budget // per_unit_cost)
    if max_units <= 0:
        return 1.0  # nadie seleccionado
    k = max(1, min(n, max_units))
    thr = np.partition(y_proba, -k)[-k]  # k-ésimo mayor
    return float(thr)


# =========================
# SHAP nativo de XGBoost (contribs e interacciones)
# =========================
def shap_contribs(
    model: XGBClassifier,
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute mean|SHAP| per feature and (if available) interaction strengths.

    Nota: usa pred_contribs/pred_interactions del booster; no requiere librería 'shap'.

    Args:
      model: XGBClassifier entrenado.
      X: Conjunto de datos para explicar (p.ej., validación).

    Returns:
      (df_importance, df_interactions)
        - df_importance: ['feature','mean_abs_shap'] ordenado desc.
        - df_interactions: ['feat_i','feat_j','mean_abs_interaction'] ordenado desc (si soportado).
    """
    import xgboost as xgb

    booster = model.get_booster()
    dmat = xgb.DMatrix(X)

    # Contribuciones SHAP (última columna es el bias)
    contribs = booster.predict(dmat, pred_contribs=True)  # (n, p+1)
    contribs = contribs[:, :-1]
    mean_abs = np.abs(contribs).mean(axis=0)
    df_imp = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )

    # Interacciones (si el build lo soporta)
    try:
        inter = booster.predict(dmat, pred_interactions=True)  # (n, p+1, p+1)
        inter = inter[:, :-1, :-1]
        mean_abs_inter = np.abs(inter).mean(axis=0)            # (p, p)
        tri_i, tri_j = np.triu_indices(mean_abs_inter.shape[0], k=1)
        df_inter = (
            pd.DataFrame({
                "feat_i": X.columns[tri_i],
                "feat_j": X.columns[tri_j],
                "mean_abs_interaction": mean_abs_inter[tri_i, tri_j],
            })
            .sort_values("mean_abs_interaction", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )
    except Exception:
        df_inter = pd.DataFrame(columns=["feat_i", "feat_j", "mean_abs_interaction"])

    return df_imp, df_inter


# =========================
# Escenarios por selección (top-k% y thresholds)
# =========================
def incentive_plans(
    y_proba: np.ndarray,
    y_true: pd.Series | np.ndarray,
    unit_cost: float = 50.0,
    topk_percents: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
    thresholds: Sequence[float] | None = None,
    beta: float = 2.0,
) -> pd.DataFrame:
    """Build selection scenarios (top-k% y thresholds) con métricas operativas.

    Args:
      y_proba: Probabilidades del modelo.
      y_true: Etiquetas reales (0/1).
      unit_cost: Coste por incentivo.
      topk_percents: Porcentajes para top-k (ej. 5%, 10%, 20%, 30%).
      thresholds: Lista de umbrales absolutos opcional.
      beta: Parámetro de Fβ (β>1 favorece recall).

    Returns:
      DataFrame con columnas:
        ['strategy','param','n_selected','cost','precision','recall','fbeta','avg_score']
    """
    y = np.asarray(y_true).astype(int)
    rows = []

    # Escenarios por top-k%
    for pct in topk_percents:
        k = max(1, int(round(len(y) * pct)))
        thr = np.partition(y_proba, -k)[-k]
        rep = threshold_metrics(y, y_proba, thr, beta=beta)
        rows.append(dict(
            strategy="topk%",
            param=pct,
            n_selected=rep.positives_pred,
            cost=rep.positives_pred * unit_cost,
            precision=rep.precision,
            recall=rep.recall,
            fbeta=rep.fbeta,
            avg_score=float(np.sort(y_proba)[-k:].mean()) if k > 0 else 0.0,
        ))

    # Escenarios por umbral absoluto
    if thresholds:
        for thr in thresholds:
            rep = threshold_metrics(y, y_proba, thr, beta=beta)
            rows.append(dict(
                strategy="threshold",
                param=thr,
                n_selected=rep.positives_pred,
                cost=rep.positives_pred * unit_cost,
                precision=rep.precision,
                recall=rep.recall,
                fbeta=rep.fbeta,
                avg_score=float(y_proba[y_proba >= thr].mean()) if rep.positives_pred > 0 else 0.0,
            ))

    return pd.DataFrame(rows).sort_values(["strategy", "param"]).reset_index(drop=True)


# =========================
# Selección de umbral con restricciones de Marketing
# =========================
def select_threshold_with_constraints(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    unit_cost: float,
    total_budget: float,
    min_precision: float = 0.2,
    beta: float = 2.0,
    grid: np.ndarray | None = None,
) -> ThresholdReport | None:
    """Elige el umbral que maximiza Fβ sujeto a coste<=presupuesto y precision>=min_precision.

    Args:
      y_true: Etiquetas reales (0/1).
      y_proba: Probabilidades.
      unit_cost: Coste por contacto/incentivo (p.ej., 50€).
      total_budget: Presupuesto máximo (p.ej., 50_000€).
      min_precision: Precision mínima aceptable (piso de pureza).
      beta: Peso del recall en Fβ.
      grid: Conjunto de umbrales a evaluar.

    Returns:
      ThresholdReport del mejor umbral que cumple restricciones, o None si no existe.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    y = np.asarray(y_true).astype(int)
    best: ThresholdReport | None = None
    for t in grid:
        rep = threshold_metrics(y, y_proba, threshold=t, beta=beta)
        cost = rep.positives_pred * unit_cost
        if cost <= total_budget and rep.precision >= min_precision:
            if best is None or rep.fbeta > best.fbeta:
                best = rep
    return best


def campaign_scenarios(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    unit_cost: float = 50.0,
    budgets: Sequence[float] = (20_000, 50_000, 100_000),
    min_precision: float = 0.2,
    beta: float = 2.0,
    benefit_per_retained: float = 200.0,  # margen/CLV incremental estimado
    uplift_rate: float = 0.10,            # % de contactados que realmente se “salvan” por el incentivo
) -> pd.DataFrame:
    """Construye planes por presupuesto con métricas y ROI esperado (aprox).

    Args:
      y_true: Etiquetas reales (0/1).
      y_proba: Probabilidades.
      unit_cost: Coste por incentivo (p.ej., 50€).
      budgets: Lista de presupuestos a simular.
      min_precision: Piso de precision.
      beta: Peso del recall en Fβ.
      benefit_per_retained: Beneficio esperado por cliente retenido (margen / CLV incremental).
      uplift_rate: Porcentaje de contactados efectivamente “salvados” por el incentivo (proxy).

    Returns:
      DataFrame con plan por presupuesto y métricas/ROI esperado.
    """
    y = np.asarray(y_true).astype(int)
    p = y_proba
    rows = []

    for B in budgets:
        # Umbral inicial por presupuesto (top-k)
        thr = select_topk_by_budget(p, per_unit_cost=unit_cost, total_budget=B)
        rep = threshold_metrics(y, p, thr, beta=beta)

        # Si no alcanzamos precision mínima, subimos umbral hasta cumplir (busca mejor Fβ dentro de la restricción)
        if rep.precision < min_precision:
            grid = np.linspace(max(0.5, thr), 0.99, 20)
            best = None
            for t in grid:
                r = threshold_metrics(y, p, t, beta=beta)
                cost = r.positives_pred * unit_cost
                if cost <= B and r.precision >= min_precision:
                    if best is None or r.fbeta > best.fbeta:
                        best = r
            if best:
                rep = best
                thr = best.threshold

        cost = rep.positives_pred * unit_cost
        expected_retained = rep.positives_pred * uplift_rate
        expected_benefit = expected_retained * benefit_per_retained
        expected_roi = (expected_benefit - cost) / cost if cost > 0 else 0.0

        rows.append(dict(
            budget=B,
            threshold=thr,
            selected=rep.positives_pred,
            cost=cost,
            precision=rep.precision,
            recall=rep.recall,
            fbeta=rep.fbeta,
            expected_retained=expected_retained,
            expected_benefit=expected_benefit,
            expected_roi=expected_roi,
        ))

    return pd.DataFrame(rows).sort_values("budget").reset_index(drop=True)




# =========================
# Visualización: ROC y PR (train vs val)
# =========================
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_curves_train_val(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series | pd.DataFrame,
    title_prefix: str = "XGB",
    show: bool = True,
) -> Dict[str, float]:
    """Plot ROC and Precision-Recall curves for train vs validation and return metrics.

    Args:
        model: Fitted XGBClassifier.
        X_train: Training features.
        y_train: Training labels (Series or single-column DataFrame).
        X_val: Validation features.
        y_val: Validation labels (Series or single-column DataFrame).
        title_prefix: Text prefix for plot titles.
        show: Whether to display the plots with matplotlib.

    Returns:
        Dict with AUC/PR-AUC for train and val and their gaps.
    """
    y_tr = to_1d_series(y_train)
    y_va = to_1d_series(y_val)

    # Probabilidades
    p_tr = model.predict_proba(X_train)[:, 1]
    p_va = model.predict_proba(X_val)[:, 1]

    # ROC
    fpr_tr, tpr_tr, _ = roc_curve(y_tr, p_tr)
    roc_auc_tr = auc(fpr_tr, tpr_tr)
    fpr_va, tpr_va, _ = roc_curve(y_va, p_va)
    roc_auc_va = auc(fpr_va, tpr_va)

    # PR
    prec_tr, rec_tr, _ = precision_recall_curve(y_tr, p_tr)
    ap_tr = average_precision_score(y_tr, p_tr)
    prec_va, rec_va, _ = precision_recall_curve(y_va, p_va)
    ap_va = average_precision_score(y_va, p_va)

    prev_tr = float(np.mean(y_tr))
    prev_va = float(np.mean(y_va))

    if show:
        # ROC
        plt.figure(figsize=(6.5, 5))
        plt.plot(fpr_tr, tpr_tr, lw=2, label=f"Train ROC (AUC = {roc_auc_tr:.3f})")
        plt.plot(fpr_va, tpr_va, lw=2, label=f"Val ROC (AUC = {roc_auc_va:.3f})")
        plt.plot([0, 1], [0, 1], ls="--", lw=1, label="Random (AUC = 0.5)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title(f"{title_prefix} — ROC (Train vs Val)")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.25)
        plt.show()

        # PR
        plt.figure(figsize=(6.5, 5))
        plt.plot(rec_tr, prec_tr, lw=2, label=f"Train PR (AP = {ap_tr:.3f})")
        plt.plot(rec_va, prec_va, lw=2, label=f"Val PR (AP = {ap_va:.3f})")
        plt.hlines(prev_tr, 0, 1, colors="gray", linestyles="--", lw=1, label=f"Base Train = {prev_tr:.3%}")
        plt.hlines(prev_va, 0, 1, colors="black", linestyles=":", lw=1, label=f"Base Val = {prev_va:.3%}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{title_prefix} — Precision–Recall (Train vs Val)")
        plt.legend(loc="upper right")
        plt.grid(alpha=0.25)
        plt.ylim(0, 1.01)
        plt.xlim(0, 1.0)
        plt.show()

    return {
        "auc_train": float(roc_auc_tr),
        "auc_val": float(roc_auc_va),
        "auc_gap": float(roc_auc_tr - roc_auc_va),
        "prauc_train": float(ap_tr),
        "prauc_val": float(ap_va),
        "prauc_gap": float(ap_tr - ap_va),
        "prev_train": prev_tr,
        "prev_val": prev_va,
    }
