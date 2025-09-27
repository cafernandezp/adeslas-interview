# src/utils/data_cleaning.py
# -*- coding: utf-8 -*-
"""
Limpieza simple y reutilizable para columnas categóricas.

Resumen rápido de utilidades (todas devuelven (serie_limpia, log_cambios) salvo que se indique):
- clean_categorical(s, synonyms=None): Normaliza a snake_case sin tildes y aplica sinónimos.
- clean_canal_entrada(s): Normalización + sinónimos comunes de canal_entrada.
- clean_cia_procedencia(s): Unifica variantes (p.ej. 'mutua_madrilenya' -> 'mutua_madrilena').
- clean_profesion(s): Corrige typos/variantes; 'desconocido' -> NaN explícito.
- clean_garantia_ampliada(s): Normaliza categorías a snake_case (p.ej. 'vehiculo_sustitucion', 'accesorios').
- clean_admite_publi(s): Sí/No a {1,0,<NA>}.
- clean_gestion_multas(s): Sí/No a {1,0,<NA>}.

Notas:
- Los logs incluyen columnas ['original', 'final', 'count'] para auditoría.
- Patrón: normalizar primero; luego mapear sinónimos/flags.
"""

from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd
from .text import to_snake_es, normalize_yesno01


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


def clean_profesion(s: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
    """Limpia 'profesion' y unifica variantes/typos mediante sinónimos.
    
    Reglas:
      - Normaliza a snake_case sin tildes (to_snake_es).
      - Corrige typos y variantes obvias:
          * 'Estdiante' → 'estudiante'
          * 'Medicina/Enfermería' y 'Medicina/Enfermeria' → 'medicina_enfermeria'
          * 'Banc0' y 'Banco' → 'banca'
          * 'Profesor/Docencia' → 'profesor_docencia'
          * 'Funcion publica' → 'funcion_publica'
          * 'Transporte de mercan' → 'transporte_de_mercancias'
          * 'Transporte de pasaje' → 'transporte_de_pasajeros'
      - 'desconocido' se convierte a NaN explícito (pd.NA).

    Args:
        s: Serie 'profesion' original.

    Returns:
        (serie_limpia, log_cambios):
            - serie_limpia: Serie normalizada y corregida.
            - log_cambios: DataFrame ['original', 'final', 'count'] para auditoría.
    """
    synonyms = {
        "estdiante": "estudiante",
        "medicina/enfermeria": "medicina_enfermeria",
        "medicina/enfermería": "medicina_enfermeria",
        "profesor/docencia": "profesor_docencia",
        "funcion publica": "funcion_publica",
        "transporte de mercan": "transporte_de_mercancias",
        "transporte de pasaje": "transporte_de_pasajeros",
        "banc0": "banca",
        "banco": "banca",
        # 'desconocido' se imputará como NaN más abajo
    }

    serie_limpia, log = clean_categorical(s, synonyms=synonyms)

    # Imputar 'desconocido' → NaN
    serie_limpia = serie_limpia.mask(serie_limpia == "desconocido", pd.NA)

    return serie_limpia, log




def clean_admite_publi(s: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Normaliza 'admite_publi' a {1,0,<NA>}."""
    mapped = s.map(normalize_yesno01).astype("Int64")
    log = (
        pd.DataFrame({"original": s.astype("string"), "final": mapped.astype("string")})
        .value_counts(["original", "final"]).rename("count").reset_index()
        .sort_values("count", ascending=False)
    )
    return mapped, log




def clean_gestion_multas(s: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Normaliza 'gestion_multas' a {1,0,<NA>} (S/Si→1, N/No→0)."""
    mapped = s.map(normalize_yesno01).astype("Int64")
    log = (
        pd.DataFrame({"original": s.astype("string"), "final": mapped.astype("string")})
        .value_counts(["original", "final"]).rename("count").reset_index()
        .sort_values("count", ascending=False)
    )
    return mapped, log



def clean_garantia_ampliada(s: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
    """Normaliza 'garantia_ampliada' a snake_case y respeta NaN.

    Ejemplos:
        'Vehiculo sustitucion' -> 'vehiculo_sustitucion'
        'Accesorios' -> 'accesorios'

    Args:
        s: Serie original 'garantia_ampliada'.

    Returns:
        (serie_limpia, log_cambios):
            - serie_limpia: valores en snake_case (mismo índice que s).
            - log_cambios: DataFrame ['original','final','count'].
    """
    # Normalización genérica reutilizando to_snake_es
    final = s.astype("string").fillna("").map(to_snake_es)

    # Log de auditoría
    log = (
        pd.DataFrame({"original": s.astype("string").fillna(""), "final": final})
        .value_counts(["original", "final"])
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )

    # Restaurar NaN donde el original era NaN
    final = final.where(s.notna(), pd.NA)
    return final, log


def clean_tipo_pago(s: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
    """Normaliza 'tipo_pago' y valida contra {'mensual','anual'}; otros -> NaN.

    Ejemplos:
        'Mensual' -> 'mensual'
        'Anual'   -> 'anual'
        'mensualidad' -> NaN (no esperado)
    """
    # 1) Normalización
    norm = s.astype("string").fillna("").map(to_snake_es)

    # 2) Validación estricta
    valid = {"mensual", "anual"}
    final = norm.where(norm.isin(valid), other=pd.NA)

    # 3) Log de auditoría
    log = (
        pd.DataFrame({"original": s.astype("string").fillna(""), "final": final.astype("string")})
        .value_counts(["original", "final"])
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )

    # 4) Restaurar NaN donde el original ya era NaN (por claridad)
    final = final.where(s.notna(), pd.NA)
    return final, log
