"""Parametros del proyecto. Los criterios de aceptacion se fijan antes de evaluar en test."""
from pathlib import Path

SEMILLA = 42
TAMANO_TEST = 0.20
PLIEGUES_CV = 5
UMBRAL_INICIAL = 0.5
# Recall objetivo al calibrar el umbral solo con predicciones fuera de pliegue de train.
RECALL_OBJETIVO_CV = 0.97

RAIZ = Path(__file__).resolve().parents[1]
DIR_RESULTADOS = RAIZ / "resultados"

# Criterios de aceptacion sobre el conjunto de test (clase positiva = maligno).
RECALL_MINIMO = 0.95
F1_MINIMO = 0.93
