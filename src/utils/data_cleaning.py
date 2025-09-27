# src/utils/data_cleaning.py
# -*- coding: utf-8 -*-
"""
Limpieza simple y reutilizable para columnas categóricas.
"""

from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd
from .text import to_snake_es


def clean_categorical(
    s: pd.Series,
    synonyms: Dict[str, str] | None = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Limpia una Serie categórica (sin tildes, minúsculas, '_' como separador) y aplica sinónimos.

    Args:
        s: Serie original (puede contener NaN).
        synonyms: Diccionario de correcciones {variante: canonico}. Tanto claves como
            valores se normalizan con las mismas reglas para robustez.

    Returns:
        (serie_limpia, log_cambios):
            - serie_limpia: Serie transformada (mismo índice que s).
            - log_cambios: DataFrame con columnas ['original', 'final', 'count'].
    """
    # 1) Normalización genérica
    norm = s.astype("string").fillna("").map(to_snake_es)

    # 2) Mapeo de sinónimos (si se pasa)
    if synonyms:
        syn_norm = {to_snake_es(k): to_snake_es(v) for k, v in synonyms.items()}
        final = norm.map(lambda x: syn_norm.get(x, x))
    else:
        final = norm

    # 3) Log muy simple
    log = (
        pd.DataFrame({"original": s.astype("string").fillna(""), "final": final})
        .value_counts(["original", "final"])
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )

    # Respeta los NaN originales
    final = final.where(s.notna(), pd.NA)
    return final, log


def clean_canal_entrada(s: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
    """Limpia 'canal_entrada' con un set mínimo de sinónimos comunes.

    Reglas incluidas (amplía según tus datos):
        - 'telfonico' → 'telefonico'
        - 'teléfonico' → 'telefonico'
        - 'telefono'   → 'telefonico'
        - 'oficina adeslas'/'oficina-adeslas' → 'oficina_adeslas'

    Args:
        s: Serie 'canal_entrada'.

    Returns:
        (serie_limpia, log_cambios)
    """
    synonyms = {
        "telfonico": "telefonico",
        "teléfonico": "telefonico",
        "telefono": "telefonico",
        "oficina adeslas": "oficina_adeslas",
        "oficina-adeslas": "oficina_adeslas",
        "oficina_adelsas": "oficina_adeslas"
    }
    return clean_categorical(s, synonyms=synonyms)


def clean_cia_procedencia(s: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
    """Limpia 'cia_procedencia' y unifica variantes conocidas vía sinónimos.

    Reglas:
      - Normaliza a snake_case sin tildes (usa to_snake_es).
      - Une 'mutua_madrilenya' y 'mutua_madrilena' en 'mutua_madrilena'.

    Args:
        s: Serie original 'cia_procedencia' (puede contener NaN).

    Returns:
        (serie_limpia, log_cambios):
            - serie_limpia: Serie transformada (mismo índice que s).
            - log_cambios: DataFrame ['original', 'final', 'count'] para auditoría.
    """
    synonyms = {
        # Caso reportado
        "mutua madrilenya": "mutua madrilena",
        "mutua madrileña": "mutua madrilena",
        # (Opcionales: ejemplos de identidad explícita o correcciones futuras)
        # "union alcoyana": "union alcoyana",
        # "pelayo": "pelayo",
        # "axa": "axa",
        # "liberty": "liberty",
        # "ocaso": "ocaso",
        # "verti": "verti",
    }
    return clean_categorical(s, synonyms=synonyms)
