"""Genera el informe PDF con tablas y gráficas del análisis ejecutado."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from src.analysis import FIGURES_DIR, RESULTS_PATH, ROOT_DIR, run_analysis

REPORT_PATH = ROOT_DIR / "reports" / "informe_segmentacion_vehiculos.pdf"


def _text_page(
    pdf: PdfPages,
    title: str,
    paragraphs: list[str],
    bullets: list[str] | None = None,
    footer: str = "",
) -> None:
    """Agrega una página de lectura clara con texto generado desde resultados."""
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(0.08, 0.94, title, fontsize=18, weight="bold", color="#174a5b")
    y = 0.88
    for paragraph in paragraphs:
        figure.text(0.08, y, paragraph, fontsize=10, va="top", wrap=True)
        y -= 0.09 if len(paragraph) < 160 else 0.13
    if bullets:
        y -= 0.02
        for bullet in bullets:
            figure.text(0.1, y, f"• {bullet}", fontsize=9.5, va="top", wrap=True)
            y -= 0.075 if len(bullet) < 150 else 0.11
    if footer:
        figure.text(0.08, 0.035, footer, fontsize=8, color="#555555")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _table_page(
    pdf: PdfPages,
    title: str,
    frame: pd.DataFrame,
    note: str = "",
    font_size: int = 8,
) -> None:
    """Agrega una tabla de estadísticas calculadas durante el análisis."""
    figure, axis = plt.subplots(figsize=(11.69, 8.27))
    axis.axis("off")
    axis.set_title(title, fontsize=16, weight="bold", loc="left", pad=20)
    if note:
        figure.text(0.08, 0.91, note, fontsize=8, color="#444444")
    table = axis.table(
        cellText=frame.values,
        colLabels=frame.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.5)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _figure_page(pdf: PdfPages, title: str, image_paths: list[Path]) -> None:
    """Incluye las gráficas creadas por el script de análisis."""
    if len(image_paths) == 1:
        figure, axis = plt.subplots(figsize=(11.69, 8.27))
        figure.suptitle(title, fontsize=16, weight="bold")
        axis.axis("off")
        image_path = image_paths[0]
        if image_path.exists():
            axis.imshow(mpimg.imread(image_path))
        else:
            axis.text(0.05, 0.5, f"Gráfica pendiente: {image_path.name}", fontsize=9)
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
        return

    figure, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    figure.suptitle(title, fontsize=16, weight="bold")
    for axis, image_path in zip(axes.flat, image_paths):
        axis.axis("off")
        if image_path.exists():
            axis.imshow(mpimg.imread(image_path))
            axis.set_title(image_path.stem.replace("_", " ").capitalize(), fontsize=9)
        else:
            axis.text(0.05, 0.5, f"Gráfica pendiente: {image_path.name}", fontsize=9)
    for axis in list(axes.flat)[len(image_paths) :]:
        axis.axis("off")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _format_profiles(results: dict[str, Any]) -> pd.DataFrame:
    profiles = pd.DataFrame(results["perfiles_clusters"])
    return pd.DataFrame(
        {
            "Clúster": [f"C{int(value) + 1}" for value in profiles["cluster"]],
            "Nombre descriptivo": profiles["nombre"],
            "Registros": profiles["registros"].map(lambda value: f"{value:,}"),
            "Participación": profiles["porcentaje"].map(lambda value: f"{value:.1f}%"),
            "Precio mediano (BDT)": profiles["precio_mediano_bdt"].map(
                lambda value: f"{value:,.0f}"
            ),
            "Edad mediana": profiles["antiguedad_mediana"].map(
                lambda value: f"{value:.1f} años"
            ),
            "Km medianos": profiles["kilometraje_mediano"].map(
                lambda value: f"{value:,.1f}"
            ),
            "Motor mediano": profiles["motor_mediano_cc"].map(
                lambda value: f"{value:,.0f} cc"
            ),
        }
    )


def generate_report() -> Path:
    """Ejecuta el análisis y crea el PDF solicitado por la rúbrica."""
    _, results = run_analysis(save_outputs=True)
    profiles = _format_profiles(results)
    centroids = pd.DataFrame(results["centroides_retransformados"])
    centroid_table = centroids[
        ["cluster", "price_bdt", "car_age", "kilometers_numeric", "engine_cc"]
    ].copy()
    centroid_table["cluster"] = centroid_table["cluster"].map(
        lambda value: f"C{int(value) + 1}"
    )
    centroid_table = centroid_table.rename(
        columns={
            "cluster": "Clúster",
            "price_bdt": "Precio BDT",
            "car_age": "Edad (años)",
            "kilometers_numeric": "Kilometraje (km)",
            "engine_cc": "Motor (cc)",
        }
    ).round(1)

    dataset = results["dataset"]
    audit = results["limpieza"]
    metrics = results["metricas_finales"]
    comparison = results["comparacion_modelos"]
    report_date = date.today().strftime("%d/%m/%Y")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(REPORT_PATH) as pdf:
        cover = plt.figure(figsize=(8.27, 11.69))
        cover.text(
            0.1,
            0.79,
            "Segmentación de vehículos usados",
            fontsize=24,
            weight="bold",
            color="#174a5b",
        )
        cover.text(
            0.1,
            0.72,
            "K-Means y dashboard interactivo en Streamlit",
            fontsize=16,
        )
        cover.text(
            0.1,
            0.61,
            "Integrantes: Jonathan Ortuño, Bryan Nuñez y Benjamin Lopez",
            fontsize=12,
        )
        cover.text(0.1, 0.56, "Grupo: 1-9 IDGS", fontsize=12)
        cover.text(0.1, 0.51, f"Fecha: {report_date}", fontsize=12)
        cover.text(
            0.1,
            0.12,
            "Proyecto escolar en equipo | Análisis exploratorio de anuncios "
            "de vehículos usados en Bangladesh",
            fontsize=10,
            color="#555555",
        )
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)

        _text_page(
            pdf,
            "Introducción y planteamiento del problema",
            [
                "Los anuncios de vehículos usados combinan precio, antigüedad, "
                "kilometraje y capacidad de motor. Explorar estas características "
                "conjuntamente puede ayudar a describir perfiles de anuncios "
                "similares sin asignar etiquetas previas.",
                "El objetivo general es identificar segmentos descriptivos de "
                "vehículos usados mediante K-Means y facilitar su exploración con "
                "un dashboard interactivo.",
            ],
            [
                "Explorar la estructura, calidad y distribución de las variables del dataset.",
                "Preparar variables numéricas comparables y evaluar cinco valores de K.",
                "Describir los grupos con estadísticas, centroides y visualizaciones.",
                "Presentar filtros, métricas y perfiles en dos secciones interactivas.",
            ],
        )

        _text_page(
            pdf,
            "Descripción del dataset y preparación",
            [
                f"El archivo contiene {dataset['registros']:,} registros y "
                f"{len(dataset['columnas_origen'])} columnas de origen. Se conserva "
                "el CSV original en la raíz del proyecto y se genera una versión "
                "procesada con variables numéricas y clúster asignado.",
                f"Fuente declarada: {dataset['fuente']}. Licencia indicada: "
                f"{dataset['licencia']}. La moneda del campo price es taka "
                "bangladesí (BDT).",
                f"Durante la validación se removieron {audit['duplicates_removed']} "
                f"duplicados exactos; antes del proceso no había valores nulos en "
                "las columnas de origen. El número de capacidades recuperadas desde "
                f"texto incrustado fue {audit['capacity_values_recovered_from_embedded_text']}.",
                "Los valores atípicos identificados mediante la regla de 1.5 IQR se "
                "conservaron: precios altos, vehículos antiguos y kilometrajes "
                "elevados pueden ser observaciones reales. Se aplicó log1p a las "
                "variables numéricas seleccionadas y después StandardScaler.",
            ],
            [
                "Variables de agrupamiento: precio BDT, antigüedad, kilometraje y capacidad del motor.",
                "Las variables originales de precio, motor y kilometraje se almacenan como texto con unidades o separadores; se parsearon sin alterar el CSV de origen.",
                "La consistencia entre año de fabricación y edad se verificó con los años implícitos calculados desde ambos campos.",
            ],
        )

        evaluation = pd.DataFrame(results["evaluacion_k"])
        evaluation_table = evaluation.rename(
            columns={
                "k": "K",
                "inertia": "Inercia",
                "silhouette": "Silhouette (muestra)",
                "davies_bouldin": "Davies-Bouldin",
            }
        ).round(4)
        _table_page(
            pdf,
            "Evaluación y selección de K",
            evaluation_table,
            (
                f"Se eligió K={results['k_seleccionado']} por obtener el mayor "
                f"silhouette entre los cinco valores evaluados. El silhouette se "
                f"estimó en hasta {results['parametros']['silhouette_sample_size']:,} "
                "registros con semilla fija; la inercia se muestra como apoyo."
            ),
        )

        comparison_frame = pd.DataFrame(
            [
                {
                    "Configuración": "Seleccionada",
                    "K": comparison["modelo_seleccionado"]["k"],
                    "Silhouette": comparison["modelo_seleccionado"]["silhouette"],
                    "Inercia": comparison["modelo_seleccionado"]["inertia"],
                    "Tamaños": str(comparison["modelo_seleccionado"]["tamanos"]),
                },
                {
                    "Configuración": "Referencia",
                    "K": comparison["modelo_referencia"]["k"],
                    "Silhouette": comparison["modelo_referencia"]["silhouette"],
                    "Inercia": comparison["modelo_referencia"]["inertia"],
                    "Tamaños": str(comparison["modelo_referencia"]["tamanos"]),
                },
            ]
        ).round(4)
        _table_page(
            pdf,
            "Comparación de dos configuraciones",
            comparison_frame,
            comparison["interpretacion"],
        )
        _table_page(
            pdf,
            "Perfiles descriptivos de los clústeres",
            profiles,
            "Los nombres se generan comparando las medianas de cada grupo con cuartiles y medianas globales.",
            font_size=7,
        )
        _table_page(
            pdf,
            "Centroides del modelo",
            centroid_table,
            "Centroides retrotransformados desde el espacio logarítmico y estandarizado; son valores geométricos aproximados, no medias aritméticas.",
            font_size=9,
        )

        _figure_page(
            pdf,
            "Gráficas exploratorias y evaluación del modelo",
            [
                FIGURES_DIR / "distribucion_precios.png",
                FIGURES_DIR / "anuncios_por_marca.png",
                FIGURES_DIR / "distribucion_antiguedad.png",
                FIGURES_DIR / "evaluacion_k.png",
            ],
        )
        _figure_page(
            pdf,
            "Visualización de la segmentación",
            [
                FIGURES_DIR / "segmentacion_kmeans.png",
                FIGURES_DIR / "segmentacion_antiguedad_precio.png",
            ],
        )
        _figure_page(
            pdf,
            "Centroides y comparación de clústeres",
            [
                FIGURES_DIR / "centroides_kmeans.png",
                FIGURES_DIR / "comparacion_clusters.png",
            ],
        )

        screenshot_paths = [
            ROOT_DIR / "reports" / "figures" / "dashboard_general.png",
            ROOT_DIR / "reports" / "figures" / "dashboard_segmentacion.png",
        ]
        for screenshot_path in screenshot_paths:
            if screenshot_path.exists():
                _figure_page(
                    pdf,
                    f"Dashboard: {screenshot_path.stem.replace('_', ' ').capitalize()}",
                    [screenshot_path],
                )
        _text_page(
            pdf,
            "Resultados, hallazgos y recomendaciones",
            [
                f"El modelo final tiene K={results['k_seleccionado']}, silhouette "
                f"{metrics['silhouette']:.4f} sobre la muestra de evaluación, "
                f"inercia {metrics['inertia']:,.2f} y Davies-Bouldin "
                f"{metrics['davies_bouldin']:.4f}. Las métricas se interpretan "
                "junto con los perfiles; ninguna por sí sola garantiza utilidad de negocio.",
            ],
            results["hallazgos"] + results["recomendaciones"],
        )
        _text_page(
            pdf,
            "Dashboard, limitaciones y conclusiones",
            [
                "El dashboard dispone de una sección de análisis general y otra de "
                "segmentación. Tres filtros (marca, transmisión y combustible) "
                "actualizan las cuatro métricas, distribuciones, dispersión y "
                "comparaciones por clúster.",
                "Los segmentos resumen similitudes en las variables elegidas; no "
                "predicen el precio ni describen el estado mecánico. Para decidir "
                "sobre un vehículo concreto se requiere una inspección individual.",
                "El dataset refleja anuncios de un mercado específico y está "
                "concentrado en algunas marcas. K-Means también es sensible a la "
                "elección de variables, la escala y el número de grupos. El precio "
                "participa en la segmentación, así que los rangos de precio de los "
                "grupos no son una validación independiente.",
                "En futuras versiones conviene agregar cobertura de marcas menos "
                "frecuentes y datos temporales de los anuncios, y contrastar la "
                "estabilidad de los perfiles con nuevas muestras.",
            ],
            footer=(
                f"Salida generada el {report_date}. Las tablas, hallazgos y gráficas "
                "se reconstruyen desde el CSV mediante los scripts del repositorio."
            ),
        )

    return REPORT_PATH


if __name__ == "__main__":
    print(f"Informe generado: {generate_report().relative_to(ROOT_DIR)}")
