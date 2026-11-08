"""Carga del dataset y separacion train/test."""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.config import SEMILLA, TAMANO_TEST


def cargar_datos():
    """Devuelve X (30 variables) e y con la clase positiva 1 = maligno.

    scikit-learn codifica 0 = maligno y 1 = benigno; aqui se invierte para que
    recall, precision y F1 se refieran a detectar el caso maligno.
    """
    crudo = load_breast_cancer(as_frame=True)
    X = crudo.data.copy()
    y = (1 - crudo.target).rename("maligno")
    return X, y


def separar(X, y):
    """Division estratificada y reproducible 80/20."""
    return train_test_split(X, y, test_size=TAMANO_TEST, stratify=y, random_state=SEMILLA)
