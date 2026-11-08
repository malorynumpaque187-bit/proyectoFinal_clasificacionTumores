"""Compara modelos con validacion cruzada en train, evalua en test y guarda evidencias.

Uso: python -m src.entrenar_evaluar
"""
import json
import math

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, make_scorer, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from src.config import (DIR_RESULTADOS, F1_MINIMO, PLIEGUES_CV, RECALL_MINIMO,
                        RECALL_OBJETIVO_CV, SEMILLA, UMBRAL_INICIAL)
from src.datos import cargar_datos, separar
from src.modelos import candidatos

METRICAS_CV = ["accuracy", "precision", "recall", "f1"]
ELEGIDO = "Regresión Logística"


def comparar_cv(X_train, y_train):
    cv = StratifiedKFold(n_splits=PLIEGUES_CV, shuffle=True, random_state=SEMILLA)
    filas = []
    for nombre, modelo in candidatos().items():
        r = cross_validate(modelo, X_train, y_train, cv=cv, scoring=_puntuadores())
        fila = {"modelo": nombre}
        for m in METRICAS_CV:
            fila[f"{m}_media"] = float(r[f"test_{m}"].mean())
            fila[f"{m}_desv"] = float(r[f"test_{m}"].std())
        filas.append(fila)
    return pd.DataFrame(filas)


def _puntuadores():
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
        "f1": make_scorer(f1_score, zero_division=0),
    }


def calibrar_umbral(modelo, X_train, y_train):
    """Mayor umbral cuyo recall fuera de pliegue alcanza el objetivo. No usa datos de test."""
    cv = StratifiedKFold(n_splits=PLIEGUES_CV, shuffle=True, random_state=SEMILLA)
    proba = cross_val_predict(modelo, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    malignos = sorted(proba[y_train.to_numpy() == 1], reverse=True)
    k = math.ceil(RECALL_OBJETIVO_CV * len(malignos))
    return float(malignos[k - 1])


def wilson(aciertos, n, z=1.96):
    """Intervalo de confianza de Wilson al 95 % para una proporcion."""
    p = aciertos / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return [float(c - m), float(c + m)]


def evaluar_criterios(m, ref):
    return {
        f"recall >= {RECALL_MINIMO}": m["recall"] >= RECALL_MINIMO,
        f"F1 >= {F1_MINIMO}": m["f1"] >= F1_MINIMO,
        "F1 supera al modelo de referencia": m["f1"] > ref["f1"],
    }


def metricas_test(modelo, X_test, y_test, umbral=UMBRAL_INICIAL):
    proba = modelo.predict_proba(X_test)[:, 1]
    pred = (proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, proba)) if len(set(proba)) > 1 else 0.5,
        "VP": int(tp), "FP": int(fp), "VN": int(tn), "FN": int(fn),
    }


def graficar_matriz(modelo, X_test, y_test, ruta, umbral):
    pred = (modelo.predict_proba(X_test)[:, 1] >= umbral).astype(int)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, labels=[1, 0], display_labels=["Maligno", "Benigno"],
        cmap="Blues", colorbar=False, ax=ax)
    ax.set_xlabel("Predicción del modelo")
    ax.set_ylabel("Diagnóstico real")
    ax.set_title(f"Matriz de confusión (test, umbral {umbral:.3f})")
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def graficar_coeficientes(modelo, columnas, ruta, n=10):
    coef = pd.Series(modelo[-1].coef_[0], index=columnas)
    top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(top.index, top.values, color=["#b5482f" if v > 0 else "#2f6db5" for v in top.values])
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("Coeficiente estandarizado (positivo empuja hacia maligno)")
    ax.set_title("Variables con mayor peso en la Regresión Logística")
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)
    return {k: float(v) for k, v in top.iloc[::-1].items()}


def main():
    DIR_RESULTADOS.mkdir(exist_ok=True)
    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = separar(X, y)

    cv = comparar_cv(X_train, y_train)
    cv.to_csv(DIR_RESULTADOS / "comparacion_cv.csv", index=False)

    ajustados = {n: clone(m).fit(X_train, y_train) for n, m in candidatos().items()}
    test = {n: metricas_test(m, X_test, y_test) for n, m in ajustados.items()}
    final = ajustados[ELEGIDO]
    train_final = metricas_test(final, X_train, y_train)
    ref = test["Referencia (clase mayoritaria)"]

    umbral = calibrar_umbral(candidatos()[ELEGIDO], X_train, y_train)
    ajustado = metricas_test(final, X_test, y_test, umbral)
    ajustado["recall_ic95"] = wilson(ajustado["VP"], ajustado["VP"] + ajustado["FN"])
    criterios_inicial = evaluar_criterios(test[ELEGIDO], ref)
    criterios_ajustado = evaluar_criterios(ajustado, ref)
    salida = {
        "dataset": {"filas": int(len(X)), "variables": int(X.shape[1]),
                    "malignos": int(y.sum()), "benignos": int((1 - y).sum()),
                    "train": int(len(X_train)), "test": int(len(X_test)),
                    "malignos_test": int(y_test.sum())},
        "modelo_elegido": ELEGIDO,
        "test_umbral_0_5": test, "train_modelo_elegido": train_final,
        "criterios_umbral_0_5": criterios_inicial,
        "aceptado_umbral_0_5": all(criterios_inicial.values()),
        "umbral_calibrado": umbral, "test_umbral_calibrado": ajustado,
        "criterios_umbral_calibrado": criterios_ajustado,
        "aceptado_umbral_calibrado": all(criterios_ajustado.values()),
    }
    graficar_matriz(final, X_test, y_test, DIR_RESULTADOS / "matriz_confusion.png", umbral)
    salida["coeficientes_top10"] = graficar_coeficientes(
        final, X.columns, DIR_RESULTADOS / "coeficientes.png")
    (DIR_RESULTADOS / "metricas.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    joblib.dump({"modelo": final, "umbral": umbral}, DIR_RESULTADOS / "modelo.joblib")
    print(cv.round(3).to_string(index=False))
    print(json.dumps(salida, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
