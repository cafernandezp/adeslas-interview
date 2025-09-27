# src/utils/text.py
# -*- coding: utf-8 -*-
"""
Utilidades de texto mínimas y claras para limpieza en español.

Funciones principales:
- remove_accents(texto): Quita tildes (NFKD).
- to_snake_es(texto, sep="_"): Pasa a snake_case minúsculas sin tildes.
- is_missing_like(x): True si el valor "parece nulo" (., -, nan, desconocido, ...).
- normalize_yesno_es(x): Devuelve 'si' | 'no' | None (para nulos/no interpretable).
- normalize_yesno01(x): Devuelve 1 | 0 | None (forma numérica).

Notas:
- Conjuntos de tokens (nulos y sí/no) están pre-normalizados para O(1) y claridad.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# -------------------- Básicas --------------------

def remove_accents(texto: str) -> str:
    """Quita tildes/diacríticos de forma determinista (NFKD)."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


_non_alnum = re.compile(r"[^a-z0-9]+")

def to_snake_es(texto: str, sep: str = "_") -> str:
    """Normaliza: strip -> sin tildes -> lower -> no alfanumérico a 'sep' -> colapsa 'sep'."""
    t = remove_accents(texto.strip()).lower()
    t = _non_alnum.sub(sep, t).strip(sep)
    t = re.sub(fr"{sep}+", sep, t)
    return t

# -------------------- Núcleo simple de normalización --------------------

def _norm(x: object) -> str:
    """str(x) seguro, sin tildes, lower y sin espacios."""
    s = "" if x is None else str(x)
    return remove_accents(s).strip().lower()

# Tokens crudos (legibles)
_NULL_TOKENS_RAW = {
    "", ".", "-", "--", "...", "na", "n/a", "nan", "null", "none",
    "sin dato", "sindato", "sd", "desconocido",
}
_YES_TOKENS_RAW = {"si", "sí", "s"}
_NO_TOKENS_RAW  = {"no", "n"}

# Tokens normalizados (uso interno; O(1) en pertenencia)
NULL_TOKENS = {_norm(t) for t in _NULL_TOKENS_RAW}
YES_TOKENS  = {_norm(t) for t in _YES_TOKENS_RAW}
NO_TOKENS   = {_norm(t) for t in _NO_TOKENS_RAW}

# -------------------- API mínima reutilizable --------------------

def is_missing_like(x: object) -> bool:
    """True si el valor 'parece nulo' (., -, nan, desconocido, ...)."""
    return _norm(x) in NULL_TOKENS


def normalize_yesno_es(x: object) -> Optional[str]:
    """Devuelve 'si' | 'no' | None (para nulos o valores no interpretables)."""
    s = _norm(x)
    if s in NULL_TOKENS:
        return None
    if s in YES_TOKENS:
        return "si"
    if s in NO_TOKENS:
        return "no"
    return None


def normalize_yesno01(x: object) -> Optional[int]:
    """Devuelve 1|0|None usando normalize_yesno_es como base."""
    v = normalize_yesno_es(x)  # 'si' | 'no' | None
    if v is None:
        return None
    return 1 if v == "si" else 0


# (Opcional) Controla lo que exporta el módulo
__all__ = [
    "remove_accents",
    "to_snake_es",
    "is_missing_like",
    "normalize_yesno_es",
    "normalize_yesno01",
    "NULL_TOKENS", "YES_TOKENS", "NO_TOKENS",
]
