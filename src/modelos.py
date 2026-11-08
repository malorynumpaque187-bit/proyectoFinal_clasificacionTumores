"""Modelos candidatos. El escalado va dentro del Pipeline para evitar fuga de datos."""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import SEMILLA


def candidatos():
    return {
        "Referencia (clase mayoritaria)": DummyClassifier(strategy="most_frequent"),
        "Regresión Logística": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEMILLA)
        ),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=4, random_state=SEMILLA),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }
