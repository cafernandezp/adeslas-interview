"""
Resumen robusto de variables numéricas y detección de outliers.

Qué hace
--------
- Calcula percentiles relevantes, IQR y umbrales de outliers por dos reglas:
  * Tukey (IQR): [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
  * MAD robusto: [mediana ± 3 * 1.4826 * MAD]
- Opcionalmente respeta "cotas factibles" por variable (p. ej., variables ≥ 0),
  usando esas cotas para el conteo de outliers (clipping de límites efectivos).
- Devuelve un DataFrame con una fila por variable para priorizar revisión/corrección.

Cuándo usarlo
-------------
- Exploración inicial (EDA) para detectar colas pesadas/outliers antes de modelar.
- Definir reglas de capping/winsorización coherentes con el dominio del dato.
- Auditoría de calidad de datos y verificación de supuestos.

Funciones y clases (resumen)
----------------------------
- OutlierBounds: contenedor de límites (tukey_low/high, mad_low/high).
- mad(x): MAD escalado (1.4826 * mediana(|x - mediana|)). Útil para robustez.
- tukey_bounds(x): devuelve (q1, q3, iqr). Base para límites de Tukey.
- compute_bounds(x): devuelve OutlierBounds combinando Tukey y MAD.
- summarize_numeric_features(df, features, ...): construye el DataFrame resumen
  con percentiles, límites (Tukey/MAD) y % fuera de límites.

Salida principal
----------------
DataFrame con columnas:
- 'feature', 'n_total', 'n_valid', 'n_nulls'
- Percentiles p00, p01, p05, p10, p25, p50, p75, p90, p95, p99, p100
- 'iqr', 'tukey_low', 'tukey_high', 'mad', 'mad_low', 'mad_high'
- 'pct_out_tukey', 'pct_out_mad'
Ordenado desc. por 'pct_out_tukey' para priorización.

Ejemplo rápido
--------------
>>> feasible = {"importe": (0.0, None)}  # variable no negativa
>>> df_sum = summarize_numeric_features(df, ["importe", "edad"], feasible_bounds=feasible)
>>> df_sum.head()

Notas
-----
- Límites teóricos (Tukey/MAD) pueden ser negativos; no es error. Para variables
  con dominio no negativo, use 'feasible_bounds' para clipear límites efectivos.
- % de outliers se calcula tras aplicar clipping (si se proporciona).
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class OutlierBounds:
    """Container for lower/upper bounds from Tukey and MAD rules."""
    tukey_low: float
    tukey_high: float
    mad_low: float
    mad_high: float


PCTS_DEFAULT: list[int] = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]


def mad(x: np.ndarray) -> float:
    """Median Absolute Deviation (scaled) for robustness.

    Args:
        x: 1-D numeric array.

    Returns:
        Scaled MAD (1.4826 * median(|x - median(x)|)).
    """
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def tukey_bounds(x: np.ndarray) -> tuple[float, float, float]:
    """Compute Tukey's IQR bounds.

    Args:
        x: 1-D numeric array.

    Returns:
        (q1, q3, iqr) where bounds are q1 - 1.5*iqr, q3 + 1.5*iqr.
    """
    q1 = np.percentile(x, 25, method="linear")
    q3 = np.percentile(x, 75, method="linear")
    iqr = q3 - q1
    return q1, q3, iqr


def compute_bounds(x: np.ndarray) -> OutlierBounds:
    """Compute Tukey and MAD bounds for a numeric array.

    Args:
        x: 1-D numeric array.

    Returns:
        OutlierBounds with Tukey and MAD cutoffs.
    """
    q1, q3, iqr = tukey_bounds(x)
    t_low = q1 - 1.5 * iqr
    t_high = q3 + 1.5 * iqr

    med = np.median(x)
    mad_val = mad(x)
    m_low = med - 3.0 * mad_val
    m_high = med + 3.0 * mad_val

    return OutlierBounds(t_low, t_high, m_low, m_high)


def summarize_numeric_features(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    percentiles: Sequence[int] = PCTS_DEFAULT,
    # Límites factibles por variable (opcional), ej.: {"importe_total": (0, None)}
    feasible_bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    replace_inf_with_nan: bool = True,
) -> pd.DataFrame:
    """Summarize percentiles & robust outlier bounds for multiple features.

    Args:
        df: Input DataFrame.
        features: Column names to summarize.
        percentiles: Percentiles in [0, 100] to compute per feature.
        feasible_bounds: Optional dict of (low, high) feasible domain limits by feature.
            If provided, lower/upper bounds (Tukey/MAD) are clipped to these limits
            for outlier-rate calculations.
        replace_inf_with_nan: If True, replaces +/-inf with NaN before dropping.

    Returns:
        DataFrame with one row per feature: counts, percentiles, Tukey/MAD bounds,
        and % outside each bound (respecting feasible clipping if provided).
    """
    rows: list[dict] = []
    pcols = [f"p{p:02d}" for p in percentiles]

    for feat in features:
        s = df[feat]
        n_total = int(s.size)

        s2 = s.replace([np.inf, -np.inf], np.nan) if replace_inf_with_nan else s
        x = s2.dropna().to_numpy()
        n_valid = int(x.size)
        n_nulls = n_total - n_valid

        row: dict[str, float | int | str] = {
            "feature": feat,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_nulls": n_nulls,
        }

        if n_valid == 0:
            for pc in pcols:
                row[pc] = np.nan
            row.update({
                "iqr": np.nan, "tukey_low": np.nan, "tukey_high": np.nan,
                "mad": np.nan, "mad_low": np.nan, "mad_high": np.nan,
                "pct_out_tukey": np.nan, "pct_out_mad": np.nan
            })
            rows.append(row)
            continue

        # Percentiles
        qs = np.percentile(x, percentiles, method="linear")
        row.update({pc: float(v) for pc, v in zip(pcols, qs)})

        # Tukey + MAD
        q1, q3, iqr = tukey_bounds(x)
        bounds = compute_bounds(x)
        med = np.median(x)
        mad_val = mad(x)

        # Cotas factibles (si existen) para calcular % outliers de forma coherente con el dominio.
        f_low, f_high = (None, None)
        if feasible_bounds and feat in feasible_bounds:
            f_low, f_high = feasible_bounds[feat]

        # Límite efectivos para el conteo de outliers (clipping a dominio factible)
        t_low_eff = bounds.tukey_low if f_low is None else max(bounds.tukey_low, f_low)
        t_high_eff = bounds.tukey_high if f_high is None else min(bounds.tukey_high, f_high)
        m_low_eff = bounds.mad_low if f_low is None else max(bounds.mad_low, f_low)
        m_high_eff = bounds.mad_high if f_high is None else min(bounds.mad_high, f_high)

        pct_out_tukey = 100.0 * ((x < t_low_eff) | (x > t_high_eff)).mean()
        pct_out_mad = 100.0 * ((x < m_low_eff) | (x > m_high_eff)).mean()

        row.update({
            "iqr": float(iqr),
            "tukey_low": float(bounds.tukey_low),
            "tukey_high": float(bounds.tukey_high),
            "mad": float(mad_val),
            "mad_low": float(bounds.mad_low),
            "mad_high": float(bounds.mad_high),
            "pct_out_tukey": float(pct_out_tukey),
            "pct_out_mad": float(pct_out_mad),
        })

        rows.append(row)

    cols = (
        ["feature", "n_total", "n_valid", "n_nulls"]
        + pcols
        + ["iqr", "tukey_low", "tukey_high", "mad", "mad_low", "mad_high",
           "pct_out_tukey", "pct_out_mad"]
    )
    out = pd.DataFrame(rows)[cols]
    return out.sort_values("pct_out_tukey", ascending=False, kind="mergesort").reset_index(drop=True)
