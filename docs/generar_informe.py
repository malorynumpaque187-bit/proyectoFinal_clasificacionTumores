"""Genera docs/informe_mini_proyecto.pdf a partir de resultados/metricas.json.

Uso: python docs/generar_informe.py
"""
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.datos import cargar_datos  # noqa: E402

M = json.loads((RAIZ / "resultados/metricas.json").read_text())
CAL = M["test_umbral_calibrado"]
BASE = M["test_umbral_0_5"]["Referencia (clase mayoritaria)"]
LR = M["test_umbral_0_5"]["Regresión Logística"]
n = lambda v, d=3: f"{v:.{d}f}".replace(".", ",")
pct = lambda v: f"{100*v:.1f} %".replace(".", ",")

e = getSampleStyleSheet()
C = ParagraphStyle("c", parent=e["BodyText"], fontName="Helvetica", fontSize=10.3, leading=14.5, spaceAfter=7)
H1 = ParagraphStyle("h1", parent=e["Heading1"], fontName="Helvetica-Bold", fontSize=13.5, spaceBefore=10, spaceAfter=6, keepWithNext=1)
H2 = ParagraphStyle("h2", parent=e["Heading2"], fontName="Helvetica-Bold", fontSize=11, spaceBefore=6, spaceAfter=4, keepWithNext=1)
T = ParagraphStyle("t", parent=C, fontSize=8.6, leading=10.8, spaceAfter=0)
TB = ParagraphStyle("tb", parent=T, fontName="Helvetica-Bold", textColor=colors.white)
CAP = ParagraphStyle("cap", parent=C, fontSize=8.8, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
REF = ParagraphStyle("r", parent=C, leftIndent=18, firstLineIndent=-18)
P = lambda t: Paragraph(t, C)


def tabla(filas, anchos):
    datos = [[Paragraph(str(c), TB if i == 0 else T) for c in f] for i, f in enumerate(filas)]
    t = Table(datos, colWidths=[a * cm for a in anchos], splitByRow=0)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa5b1")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]))
    return t


def bullets(items):
    return [Paragraph(f"•&nbsp;&nbsp;{i}", ParagraphStyle("b", parent=C, leftIndent=14, firstLineIndent=-12, spaceAfter=4)) for i in items]


X, y = cargar_datos()
cols = ["mean radius", "mean texture", "mean area", "mean concavity", "worst radius"]
muestra = [["Fila"] + cols + ["maligno"]]
for i in list(X.index[y == 1][:3]) + list(X.index[y == 0][:3]):
    muestra.append([i] + [n(X.loc[i, c], 2) for c in cols] + [int(y[i])])

h = []
h += [Spacer(1, 3.5 * cm),
      Paragraph("Mini-proyecto de Machine Learning", ParagraphStyle("tt", parent=H1, fontSize=21, leading=26, alignment=TA_CENTER)),
      Spacer(1, .3 * cm),
      Paragraph("Clasificación de tumores de mama con Regresión Logística: desempeño y análisis ético", ParagraphStyle("st", parent=C, fontSize=13, leading=18, alignment=TA_CENTER)),
      Spacer(1, 2.2 * cm)]
for l in ["<b>Autor: Malory Numpaque</b>", "Teoría de la Computación II", "Actividad 6. Mini-proyecto", "Universidad de La Salle", "Noviembre de 2026"]:
    h.append(Paragraph(l, ParagraphStyle("p", parent=C, fontSize=11.5, alignment=TA_CENTER)))
URL_REPO = "https://github.com/malorynumpaque187-bit/proyectoFinal_clasificacionTumores.git"
h += [Spacer(1, 1.2 * cm), Paragraph("<b>Repositorio:</b>", ParagraphStyle("p", parent=C, fontSize=10.5, alignment=TA_CENTER)),
      Paragraph(f'<link href="{URL_REPO}" color="#1f4e9c"><u>{URL_REPO}</u></link>', ParagraphStyle("p", parent=C, fontSize=9.5, alignment=TA_CENTER))]
h.append(PageBreak())

h += [Paragraph("1. Selección del modelo", H1),
      P("Se eligió la <b>Regresión Logística</b> para clasificar tumores como malignos o benignos. El problema tiene una salida binaria y la Regresión Logística entrega una probabilidad, lo que permite ajustar el umbral según el costo de cada error. Además, sus coeficientes permiten revisar qué variables influyen en la predicción, algo importante en un contexto de salud."),
      P(f"La elección se comparó con un modelo de referencia, un Árbol de Decisión y KNN mediante validación cruzada de 5 pliegues sobre entrenamiento. La Regresión Logística obtuvo los mayores valores medios de recall (0,953) y F1 (0,964)."),
      Paragraph("2. Dataset utilizado", H1),
      P(f"Se usó <b>Breast Cancer Wisconsin (Diagnostic)</b>, incluido en scikit-learn y procedente del UCI Machine Learning Repository (Wolberg et al., 1993). Tiene {M['dataset']['filas']} registros: {M['dataset']['malignos']} malignos y {M['dataset']['benignos']} benignos, sin valores faltantes. Las variables se calculan sobre imágenes digitalizadas de aspiración con aguja fina (FNA) de masas mamarias."),
      P("<b>Variables.</b> Son 30 numéricas: para cada una de 10 características del núcleo celular (radio, textura, perímetro, área, suavidad, compacidad, concavidad, puntos cóncavos, simetría y dimensión fractal) se registran la media, el error estándar y el peor valor."),
      P("<b>Objetivo.</b> Predecir la variable <i>maligno</i> (1 = maligno, 0 = benigno). scikit-learn usa la codificación contraria, por lo que se invirtió para que las métricas midan la detección del caso maligno."),
      tabla(muestra, [1.1, 2.6, 2.6, 2.4, 2.8, 2.6, 2.0]),
      Paragraph("Tabla 1. Muestra de 5 de las 30 variables para seis registros reales del dataset.", CAP),
      P(f"Los datos se dividieron de forma estratificada en {M['dataset']['train']} registros de entrenamiento y {M['dataset']['test']} de prueba ({M['dataset']['malignos_test']} malignos), con semilla fija."),
      Paragraph("3. Métricas de evaluación", H1),
      P("Primero se evaluó el modelo con el umbral habitual de 0,5, y se obtuvo un recall de 0,929, inferior al criterio de 0,95 fijado antes de la evaluación. Por eso se rechazó esa configuración y se calibró el umbral usando solo predicciones fuera de pliegue de entrenamiento: se tomó el mayor umbral con recall de al menos 0,97, que resultó ser <b>" + n(M["umbral_calibrado"]) + "</b>."),
      tabla([["Modelo en prueba", "Accuracy", "Precision", "Recall", "F1"],
             ["Referencia (clase mayoritaria)", n(BASE["accuracy"]), n(BASE["precision"]), n(BASE["recall"]), n(BASE["f1"])],
             ["Regresión Logística, umbral 0,5", n(LR["accuracy"]), n(LR["precision"]), n(LR["recall"]), n(LR["f1"])],
             ["<b>Regresión Logística, umbral " + n(M["umbral_calibrado"]) + "</b>", n(CAL["accuracy"]), n(CAL["precision"]), n(CAL["recall"]), n(CAL["f1"])]],
            [7.5, 2.2, 2.2, 2.2, 2.2]),
      Paragraph("Tabla 2. Resultados en el conjunto de prueba.", CAP),
      Paragraph("Interpretación del modelo final", H2),
      tabla([["Métrica", "Valor", "Interpretación"],
             ["Accuracy", pct(CAL["accuracy"]), f"El modelo acertó {CAL['VP']+CAL['VN']} de {M['dataset']['test']} casos. Por sí sola es insuficiente: el modelo de referencia logra {pct(BASE['accuracy'])} sin detectar ningún maligno."],
             ["Precision", pct(CAL["precision"]), f"De {CAL['VP']+CAL['FP']} casos señalados como malignos, {CAL['VP']} lo eran."],
             ["Recall", pct(CAL["recall"]), f"Detectó {CAL['VP']} de {CAL['VP']+CAL['FN']} tumores malignos. Es la métrica prioritaria, porque un maligno no detectado es el error más grave."],
             ["F1-Score", pct(CAL["f1"]), "Resume el equilibrio entre precision y recall."]],
            [2.4, 1.9, 12]),
      Spacer(1, 8),
      P(f"<b>Validez e incertidumbre.</b> Con solo {CAL['VP']+CAL['FN']} malignos, el intervalo de confianza del 95 % (Wilson) para el recall va de {pct(CAL['recall_ic95'][0])} a {pct(CAL['recall_ic95'][1])}. Además, el conjunto de prueba ya se había consultado con el umbral de 0,5, así que la medición final no es completamente independiente, aunque el umbral no se calculó con esos datos."),
      Paragraph("4. Matriz de confusión", H1),
      Image(str(RAIZ / "resultados/matriz_confusion.png"), width=8.2 * cm, height=7.13 * cm),
      Paragraph("Figura 1. Matriz de confusión del modelo final en prueba.", CAP),
      tabla([["Celda", "Cantidad", "Significado en este problema"],
             ["Verdaderos positivos (VP)", CAL["VP"], "Tumores malignos identificados correctamente como malignos."],
             ["Falsos positivos (FP)", CAL["FP"], "Tumor benigno señalado como maligno. Puede causar ansiedad y exámenes adicionales."],
             ["Verdaderos negativos (VN)", CAL["VN"], "Tumores benignos identificados correctamente como benignos."],
             ["Falsos negativos (FN)", CAL["FN"], "Tumores malignos clasificados como benignos. Es el error más grave, porque puede retrasar el tratamiento."]],
            [4.6, 1.8, 9.9]),
      Paragraph("5. Análisis ético", H1),
      Paragraph("Sesgos del modelo y del dataset", H2),
      P("Los datos provienen de los hospitales de la Universidad de Wisconsin, en Madison, y fueron donados en 1995. Una sola institución y época pueden no representar a otras poblaciones, equipos de imagen o protocolos de laboratorio. El etiquetado y la selección de pacientes también pueden introducir sesgos, porque solo incluye casos que llegaron a este procedimiento. Obermeyer et al. (2019) muestran que un algoritmo de salud puede producir grandes sesgos aunque parezca preciso."),
      Paragraph("Equidad", H2),
      P("El dataset no contiene edad, etnia, nivel socioeconómico ni otras variables de grupo. Por tanto, <b>no fue posible medir la equidad</b> del modelo entre poblaciones: los resultados globales no demuestran que funcione igual para todas. Antes de cualquier uso real haría falta comparar recall y falsos negativos por subgrupo."),
      Paragraph("Privacidad y uso responsable de los datos", H2),
      P("El dataset público no incluye nombres, y la versión usada omite el identificador de la muestra. Aun así, un sistema real trataría datos de salud, que en Colombia son datos sensibles según el artículo 5 de la Ley 1581 de 2012; el artículo 6 restringe su tratamiento y contempla, entre otras condiciones, la autorización explícita del titular. Se usó el dataset conforme a su licencia CC BY 4.0, con atribución y solo con fines educativos."),
      Paragraph("Impacto social", H2),
      P("Un sistema validado podría priorizar revisiones y apoyar a equipos con poca disponibilidad de especialistas. También podría causar daño: los falsos negativos retrasan tratamientos, los falsos positivos generan angustia y costos, y una confianza excesiva puede debilitar el criterio clínico. El acceso desigual a esta tecnología podría ampliar brechas existentes."),
      Paragraph("Justicia y transparencia en la toma de decisiones", H2),
      P("La recomendación de la UNESCO (2021) incluye la equidad y no discriminación, la supervisión y decisión humanas, y la transparencia y explicabilidad. En este proyecto, el modelo debe limitarse a ser un apoyo y la decisión debe permanecer en un profesional de salud. Los coeficientes de la Figura 2 aportan transparencia, pero deben leerse con cautela: varias variables están relacionadas entre sí y los coeficientes no indican causalidad. Toda persona afectada debería saber que se usó un modelo y poder solicitar una revisión humana."),
      Image(str(RAIZ / "resultados/coeficientes.png"), width=12.6 * cm, height=6.84 * cm),
      Paragraph("Figura 2. Variables con mayor peso en el modelo.", CAP),
      Paragraph("6. Conclusiones técnicas", H1)]
h += bullets([f"El modelo funcionó bien en este dataset: F1 de {n(CAL['f1'])} y recall de {n(CAL['recall'])} en prueba, muy por encima del modelo de referencia.",
              "Con el umbral de 0,5 no cumplió el criterio de recall. Calibrar el umbral permitió reducir los falsos negativos de 3 a 2 sin aumentar los falsos positivos en prueba.",
              "La diferencia entre entrenamiento (F1 de 0,982) y prueba (0,951) con umbral 0,5 es moderada, y no sugiere un sobreajuste fuerte.",
              "El buen resultado no demuestra utilidad clínica: la muestra es pequeña, proviene de una sola fuente y falta validación externa."])
h += [Paragraph("7. Recomendaciones para mejorar", H1)]
h += bullets(["Validar el modelo con datos externos de otras instituciones y en un conjunto de prueba nuevo, usado una sola vez.",
              "Incluir variables demográficas, con las debidas garantías legales, para auditar la equidad por subgrupos.",
              "Probar la reducción de variables correlacionadas y buscar hiperparámetros con validación cruzada anidada.",
              "Definir con especialistas clínicos el costo aceptable de falsos negativos y positivos para fijar el umbral.",
              "Calibrar las probabilidades y monitorear el desempeño si en el futuro el modelo se usara con datos nuevos."])
h += [Paragraph("Referencias", H1)]
for r in ["Congreso de Colombia. (2012). <i>Ley Estatutaria 1581 de 2012, por la cual se dictan disposiciones generales para la protección de datos personales</i>.",
          "Obermeyer, Z., Powers, B., Vogeli, C. y Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. <i>Science, 366</i>(6464), 447-453. https://doi.org/10.1126/science.aax2342",
          "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. <i>Journal of Machine Learning Research, 12</i>, 2825-2830.",
          "Street, W. N., Wolberg, W. H. y Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. <i>Biomedical Image Processing and Biomedical Visualization, 1905</i>, 861-870. SPIE.",
          "UNESCO. (2021). <i>Recomendación sobre la ética de la inteligencia artificial</i>. https://unesdoc.unesco.org/ark:/48223/pf0000381137",
          "Wolberg, W., Mangasarian, O., Street, N. y Street, W. (1993). <i>Breast Cancer Wisconsin (Diagnostic)</i> [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B"]:
    h.append(Paragraph(r, REF))


def pie(c, d):
    if d.page > 1:
        c.setFont("Helvetica", 8.5); c.setFillColor(colors.HexColor("#555555"))
        c.drawString(2.5 * cm, 1.3 * cm, "Autor: Malory Numpaque")
        c.drawRightString(letter[0] - 2.5 * cm, 1.3 * cm, str(d.page))


SimpleDocTemplate(str(RAIZ / "docs/informe_mini_proyecto.pdf"), pagesize=letter, leftMargin=2.5 * cm,
                  rightMargin=2.5 * cm, topMargin=2.2 * cm, bottomMargin=2.2 * cm,
                  title="Mini-proyecto de Machine Learning", author="Malory Numpaque").build(h, onFirstPage=pie, onLaterPages=pie)
