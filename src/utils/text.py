# src/utils/text.py
# -*- coding: utf-8 -*-
"""
Funciones de texto muy simples para limpieza en español.
"""

from __future__ import annotations
import unicodedata
import re


def remove_accents(texto: str) -> str:
    """Quita tildes/diacríticos de forma determinista (NFKD).

    Args:
        texto: Cadena de entrada (puede tener tildes).

    Returns:
        Cadena sin tildes.
    """
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


_non_alnum = re.compile(r"[^a-z0-9]+")

def to_snake_es(texto: str, sep: str = "_") -> str:
    """Normaliza texto para variables categóricas en español.

    Reglas: strip → quitar tildes → minúsculas → no alfanumérico a '_' → colapsar '_'.

    Args:
        texto: Cadena a normalizar.
        sep: Separador a usar para los reemplazos.

    Returns:
        Cadena en minúsculas, sin tildes y con separadores unificados.
    """
    t = remove_accents(texto.strip()).lower()
    t = _non_alnum.sub(sep, t).strip(sep)
    t = re.sub(fr"{sep}+", sep, t)
    return t
