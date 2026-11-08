"""Clasifica registros de un CSV con las 30 columnas del dataset (solo uso academico).

Uso: python -m src.predecir entrada.csv salida.csv
"""
import sys

import joblib
import pandas as pd

from src.config import DIR_RESULTADOS


def predecir(df, artefacto=None):
    artefacto = artefacto or joblib.load(DIR_RESULTADOS / "modelo.joblib")
    modelo, umbral = artefacto["modelo"], artefacto["umbral"]
    esperadas = list(modelo.feature_names_in_)
    faltantes = [c for c in esperadas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}")
    if df[esperadas].isna().any().any():
        raise ValueError("Hay valores faltantes; el sistema no imputa datos.")
    proba = modelo.predict_proba(df[esperadas])[:, 1]
    return pd.DataFrame({"prob_maligno": proba,
                         "prediccion": ["maligno" if p >= umbral else "benigno" for p in proba]})


if __name__ == "__main__":
    predecir(pd.read_csv(sys.argv[1])).to_csv(sys.argv[2], index=False)
