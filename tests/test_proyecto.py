"""Pruebas de datos, reproducibilidad, entrada y criterios de aceptacion."""
import json

import joblib
import pandas as pd
import pytest

from src.config import DIR_RESULTADOS
from src.datos import cargar_datos, separar
from src.predecir import predecir


@pytest.fixture(scope="module")
def particion():
    X, y = cargar_datos()
    return X, y, separar(X, y)


def test_clase_positiva_es_maligno(particion):
    _, y, _ = particion
    assert y.sum() == 212 and (1 - y).sum() == 357


def test_sin_fuga_de_indices(particion):
    _, _, (X_tr, X_te, _, _) = particion
    assert set(X_tr.index).isdisjoint(X_te.index)


def test_sin_filas_duplicadas_entre_conjuntos(particion):
    _, _, (X_tr, X_te, _, _) = particion
    assert len(X_tr.merge(X_te, how="inner")) == 0


def test_estratificacion(particion):
    _, y, (_, _, y_tr, y_te) = particion
    assert abs(y_tr.mean() - y.mean()) < 0.01 and abs(y_te.mean() - y.mean()) < 0.01


def test_division_reproducible():
    X, y = cargar_datos()
    a, b = separar(X, y)[1], separar(X, y)[1]
    pd.testing.assert_frame_equal(a, b)


def test_rechaza_columnas_faltantes():
    with pytest.raises(ValueError):
        predecir(pd.DataFrame({"mean radius": [1.0]}))


def test_rechaza_valores_faltantes(particion):
    X, _, _ = particion
    fila = X.head(1).copy()
    fila.iloc[0, 0] = None
    with pytest.raises(ValueError):
        predecir(fila)


def test_prediccion_valida(particion):
    X, _, _ = particion
    r = predecir(X.head(5))
    assert r["prob_maligno"].between(0, 1).all()
    assert set(r["prediccion"]) <= {"maligno", "benigno"}


def test_criterios_de_aceptacion_del_modelo_final():
    m = json.loads((DIR_RESULTADOS / "metricas.json").read_text())
    assert m["aceptado_umbral_calibrado"], m["criterios_umbral_calibrado"]
