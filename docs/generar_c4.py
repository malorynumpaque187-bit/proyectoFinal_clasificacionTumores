"""Genera el diagrama C4 de contexto (docs/c4_contexto.png)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

AZUL_OSC, AZUL, GRIS = "#0b3d73", "#2b6cb0", "#7a7f87"
fig, ax = plt.subplots(figsize=(11, 6.6))
ax.set_xlim(0, 110); ax.set_ylim(0, 66); ax.axis("off")


def caja(x, y, w, h, color, titulo, tipo, desc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                                fc=color, ec=color))
    cx = x + w / 2
    ax.text(cx, y + h - 3, titulo, color="white", ha="center", va="center", fontsize=11.5, weight="bold")
    ax.text(cx, y + h - 6.2, tipo, color="white", ha="center", va="center", fontsize=8.5)
    ax.text(cx, y + (h - 7.5) / 2, desc, color="white", ha="center", va="center", fontsize=9, linespacing=1.3)


def flecha(a, b, texto, pos, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color="#444",
                                 lw=1.2, connectionstyle=f"arc3,rad={rad}", ls="--"))
    ax.text(pos[0], pos[1], texto, fontsize=8.8, ha="center", va="center", color="#222",
            bbox=dict(fc="white", ec="none", pad=1.5))


ax.text(55, 64, "Diagrama C4 de contexto: Clasificador de tumores (FNA)", ha="center", fontsize=13.5, weight="bold")
caja(4, 44, 30, 15, AZUL_OSC, "Estudiante / Analista", "[Persona]", "Entrena, evalúa y clasifica\nregistros de prueba")
caja(76, 44, 30, 15, AZUL_OSC, "Docente evaluador", "[Persona]", "Revisa rama develop,\nmétricas y documentación")
caja(36, 22, 38, 17, AZUL, "Clasificador de tumores", "[Sistema de software]", "Regresión Logística en Python: compara\nmodelos, evalúa y genera predicciones")
caja(4, 3, 32, 13, GRIS, "scikit-learn datasets", "[Sistema externo]", "WDBC (origen: UCI ML Repository)")
caja(74, 3, 32, 13, GRIS, "GitHub", "[Sistema externo]", "Código, ramas, resultados y docs")
flecha((26, 43.5), (42, 39.5), "ejecuta por línea de comandos", (24, 41))
flecha((84, 43.5), (68, 39.5), "consulta metricas.json e informe", (87, 41))
flecha((44, 21.5), (24, 16.5), "carga 569 registros etiquetados", (24, 19.5))
flecha((66, 21.5), (86, 16.5), "versiona código y evidencias (git push)", (88, 19.5))
fig.tight_layout()
fig.savefig(Path(__file__).parent / "c4_contexto.png", dpi=170)
