"""Preparación, evaluación y segmentación reproducible del dataset de vehículos."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "UsedCarInfo.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "used_cars_clustered.csv"
RESULTS_PATH = ROOT_DIR / "reports" / "analysis_results.json"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
SOURCE_COLUMNS = [
    "title",
    "brand",
    "model",
    "transmission",
    "fuel_type",
    "price",
    "year_of_manufacture",
    "energy_capacity",
    "kilometers_run",
    "car_age",
]
FEATURES = ["price_bdt", "car_age", "kilometers_numeric", "engine_cc"]
K_VALUES = range(2, 7)
RANDOM_STATE = 42
SILHOUETTE_SAMPLE_SIZE = 2000


def _parse_formatted_number(values: pd.Series, column_name: str) -> pd.Series:
    """Convierte texto con unidades o separadores en valores numéricos."""
    extracted = values.astype("string").str.replace(r"[^0-9.]", "", regex=True)
    parsed = pd.to_numeric(extracted.replace("", pd.NA), errors="coerce")
    if parsed.isna().any():
        examples = values.loc[parsed.isna()].astype(str).head(5).tolist()
        raise ValueError(
            f"No se pudieron convertir todos los valores de {column_name}: {examples}"
        )
    return parsed.astype(float)


def prepare_data(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Valida, limpia y transforma los campos numéricos del CSV original."""
    missing_columns = sorted(set(SOURCE_COLUMNS) - set(source.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {missing_columns}")

    original_rows = len(source)
    missing_before = source[SOURCE_COLUMNS].isna().sum().astype(int).to_dict()
    cleaned = source.drop_duplicates().copy()
    duplicates_removed = original_rows - len(cleaned)

    # La expresión busca el número junto a la unidad, incluso si hay texto antes.
    capacity_text = cleaned["energy_capacity"].astype("string")
    engine_text = capacity_text.str.extract(
        r"([\d,]+(?:\.\d+)?)\s*cc\b", flags=re.IGNORECASE, expand=False
    )
    cleaned["engine_cc"] = _parse_formatted_number(
        engine_text, "energy_capacity (cc)"
    )
    cleaned["price_bdt"] = _parse_formatted_number(cleaned["price"], "price")
    cleaned["kilometers_numeric"] = _parse_formatted_number(
        cleaned["kilometers_run"], "kilometers_run"
    )
    cleaned["year_of_manufacture"] = pd.to_numeric(
        cleaned["year_of_manufacture"], errors="coerce"
    )
    cleaned["car_age"] = pd.to_numeric(cleaned["car_age"], errors="coerce")

    numeric_features = cleaned[FEATURES]
    if numeric_features.isna().any().any():
        invalid = numeric_features.isna().sum()
        raise ValueError(f"Hay valores faltantes en variables del modelo:\n{invalid}")
    if not np.isfinite(numeric_features.to_numpy(dtype=float)).all():
        raise ValueError("El dataset contiene valores no finitos en variables numéricas.")
    if (cleaned[["price_bdt", "engine_cc"]] <= 0).any().any():
        raise ValueError("Precio y capacidad del motor deben ser mayores que cero.")
    if (cleaned[["car_age", "kilometers_numeric"]] < 0).any().any():
        raise ValueError("Edad y kilometraje no pueden ser negativos.")
    if cleaned[SOURCE_COLUMNS].isna().any().any():
        raise ValueError(
            "Hay valores nulos en columnas de origen; revise el CSV antes de modelar."
        )

    outliers_iqr: dict[str, int] = {}
    for column in FEATURES:
        first_quartile, third_quartile = cleaned[column].quantile([0.25, 0.75])
        iqr = third_quartile - first_quartile
        lower_bound = first_quartile - 1.5 * iqr
        upper_bound = third_quartile + 1.5 * iqr
        outliers_iqr[column] = int(
            ((cleaned[column] < lower_bound) | (cleaned[column] > upper_bound)).sum()
        )

    ordinary_capacity = capacity_text.str.fullmatch(
        r"\s*[\d,]+(?:\.\d+)?\s*cc\s*", case=False, na=False
    )
    audit = {
        "rows_original": int(original_rows),
        "rows_after_cleaning": int(len(cleaned)),
        "columns_original": int(len(SOURCE_COLUMNS)),
        "duplicates_removed": int(duplicates_removed),
        "missing_values_before": missing_before,
        "missing_values_after": {
            column: int(count)
            for column, count in cleaned[SOURCE_COLUMNS].isna().sum().items()
        },
        "capacity_values_recovered_from_embedded_text": int((~ordinary_capacity).sum()),
        "iqr_outlier_counts_retained": outliers_iqr,
        "implied_listing_years": sorted(
            (cleaned["year_of_manufacture"] + cleaned["car_age"])
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
    }
    return cleaned.reset_index(drop=True), audit


def _cluster_name(
    price_median: float,
    age_median: float,
    mileage_median: float,
    global_price_quantiles: tuple[float, float],
    global_age_median: float,
    global_mileage_median: float,
    cluster_number: int,
) -> str:
    """Genera una etiqueta descriptiva a partir de las medianas observadas."""
    price_q1, price_q3 = global_price_quantiles
    if price_median <= price_q1:
        price_label = "Precio bajo"
    elif price_median >= price_q3:
        price_label = "Precio alto"
    else:
        price_label = "Precio intermedio"

    age_label = "más recientes" if age_median <= global_age_median else "mayor antigüedad"
    mileage_label = (
        "menor recorrido"
        if mileage_median <= global_mileage_median
        else "mayor recorrido"
    )
    return f"{price_label}, {age_label} y {mileage_label} (C{cluster_number})"


def _make_figures(
    data: pd.DataFrame,
    evaluation: list[dict[str, Any]],
    selected_k: int,
    centroids: pd.DataFrame | None = None,
) -> None:
    """Guarda gráficas exploratorias y de evaluación construidas desde el CSV."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    figure, axis = plt.subplots(figsize=(9, 5))
    sns.histplot(data=data, x="price_bdt", bins=40, ax=axis, color="#277da1")
    axis.set_xscale("log")
    axis.set_title("Distribución de precios publicados (eje logarítmico)")
    axis.set_xlabel("Precio en BDT")
    axis.set_ylabel("Número de vehículos")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "distribucion_precios.png", dpi=160)
    plt.close(figure)

    top_brands = data["brand"].value_counts().head(10).index
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.countplot(
        data=data[data["brand"].isin(top_brands)],
        y="brand",
        order=data["brand"].value_counts().head(10).index,
        ax=axis,
        color="#43aa8b",
    )
    axis.set_title("Marcas con más anuncios")
    axis.set_xlabel("Número de vehículos")
    axis.set_ylabel("Marca")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "anuncios_por_marca.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    sns.histplot(data=data, x="car_age", bins=30, ax=axis, color="#f8961e")
    axis.set_title("Distribución de la antigüedad de los vehículos")
    axis.set_xlabel("Antigüedad declarada (años)")
    axis.set_ylabel("Número de vehículos")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "distribucion_antiguedad.png", dpi=160)
    plt.close(figure)

    evaluation_frame = pd.DataFrame(evaluation)
    figure, left_axis = plt.subplots(figsize=(9, 5))
    left_axis.plot(
        evaluation_frame["k"],
        evaluation_frame["inertia"],
        marker="o",
        color="#277da1",
        label="Inercia",
    )
    left_axis.set_xlabel("Número de clústeres (K)")
    left_axis.set_ylabel("Inercia")
    right_axis = left_axis.twinx()
    right_axis.plot(
        evaluation_frame["k"],
        evaluation_frame["silhouette"],
        marker="s",
        color="#f94144",
        label="Silhouette",
    )
    right_axis.set_ylabel("Silhouette (muestra)")
    left_axis.axvline(selected_k, color="#555555", linestyle="--", alpha=0.7)
    left_axis.set_title("Evaluación de K = 2 a 6")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "evaluacion_k.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=data,
        x="kilometers_numeric",
        y="price_bdt",
        hue="cluster_name",
        alpha=0.7,
        s=30,
        ax=axis,
    )
    axis.set_xscale("symlog", linthresh=1)
    axis.set_yscale("log")
    axis.set_title("Segmentos K-Means: precio y kilometraje")
    axis.set_xlabel("Kilometraje (escala simétrica logarítmica)")
    axis.set_ylabel("Precio en BDT (escala logarítmica)")
    axis.legend(title="Clúster", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "segmentacion_kmeans.png", dpi=160)
    plt.close(figure)

    segment_codes = data["cluster"].map(lambda value: f"C{int(value) + 1}")
    order = sorted(segment_codes.unique())
    palette = dict(zip(order, sns.color_palette("deep", len(order))))
    plot_data = data.assign(segmento=segment_codes)

    figure, axis = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=plot_data.sort_values("segmento"),
        x="car_age",
        y="price_bdt",
        hue="segmento",
        hue_order=order,
        palette=palette,
        alpha=0.6,
        s=18,
        edgecolor=None,
        ax=axis,
    )
    axis.set_yscale("log")
    axis.set_title("Segmentos K-Means: antigüedad y precio")
    axis.set_xlabel("Antigüedad (años)")
    axis.set_ylabel("Precio en BDT (escala logarítmica)")
    axis.legend(title="Clúster")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "segmentacion_antiguedad_precio.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 4, figsize=(13, 4))
    box_specs = [
        ("price_bdt", "Precio (BDT, log)"),
        ("car_age", "Antigüedad (años)"),
        ("kilometers_numeric", "Kilometraje (km)"),
        ("engine_cc", "Motor (cc)"),
    ]
    for axis, (column, title) in zip(axes, box_specs):
        sns.boxplot(
            data=plot_data,
            x="segmento",
            y=column,
            hue="segmento",
            order=order,
            hue_order=order,
            palette=palette,
            showfliers=False,
            legend=False,
            ax=axis,
        )
        if column == "price_bdt":
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("")
    figure.suptitle("Comparación de las variables de agrupamiento por clúster (sin atípicos)")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "comparacion_clusters.png", dpi=160)
    plt.close(figure)

    if centroids is not None:
        labels = {
            "price_bdt": "Precio (BDT)",
            "car_age": "Antigüedad (años)",
            "kilometers_numeric": "Kilometraje (km)",
            "engine_cc": "Motor (cc)",
        }
        values = centroids.set_index(
            centroids["cluster"].map(lambda value: f"C{int(value) + 1}")
        )[FEATURES]
        # El color compara cada centroide con la mediana global de la variable.
        relative = values / data[FEATURES].median() * 100
        annotations = values.apply(
            lambda column: column.map(
                (lambda value: f"{value:.1f}")
                if column.name == "car_age"
                else (lambda value: f"{value:,.0f}")
            )
        )
        figure, axis = plt.subplots(figsize=(9, 3.6))
        sns.heatmap(
            relative.rename(columns=labels),
            annot=annotations.rename(columns=labels),
            fmt="",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=0, vcenter=100, vmax=300),
            linewidths=1,
            cbar_kws={"label": "% respecto a la mediana global"},
            ax=axis,
        )
        axis.set_title("Centroides K-Means (valor real; color = % de la mediana global)")
        axis.set_xlabel("")
        axis.set_ylabel("Clúster")
        axis.tick_params(axis="y", rotation=0)
        figure.tight_layout()
        figure.savefig(FIGURES_DIR / "centroides_kmeans.png", dpi=160)
        plt.close(figure)


def _build_summary(
    data: pd.DataFrame,
    audit: dict[str, Any],
    evaluation: list[dict[str, Any]],
    selected_k: int,
    cluster_profiles: pd.DataFrame,
    centroids: pd.DataFrame,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    """Construye resultados y narrativas a partir de estadísticas calculadas."""
    profile_rows = cluster_profiles.to_dict(orient="records")
    centroid_rows = centroids.to_dict(orient="records")

    largest = max(profile_rows, key=lambda row: row["registros"])
    smallest = min(profile_rows, key=lambda row: row["registros"])
    price_low = min(profile_rows, key=lambda row: row["precio_mediano_bdt"])
    price_high = max(profile_rows, key=lambda row: row["precio_mediano_bdt"])
    age_global_median = float(data["car_age"].median())
    mileage_global_median = float(data["kilometers_numeric"].median())

    findings = [
        (
            f"El grupo más numeroso es {largest['nombre']}: "
            f"{largest['registros']:,} vehículos "
            f"({largest['porcentaje']:.1f}% del dataset). El menor es "
            f"{smallest['nombre']} con {smallest['registros']:,} "
            f"({smallest['porcentaje']:.1f}%)."
        ),
        (
            f"Las medianas de precio van de {price_low['precio_mediano_bdt']:,.0f} BDT "
            f"en {price_low['nombre']} a {price_high['precio_mediano_bdt']:,.0f} BDT "
            f"en {price_high['nombre']}."
        ),
        (
            f"La mediana global es de {age_global_median:.1f} años de antigüedad y "
            f"{mileage_global_median:,.0f} km; los perfiles por clúster permiten "
            "comparar su combinación con el precio."
        ),
    ]
    recommendations = [
        (
            f"Usar los rangos de precio observados por grupo como referencia descriptiva "
            f"al explorar anuncios similares a {price_low['nombre']} y "
            f"{price_high['nombre']}; no tratarlos como una tasación."
        ),
        (
            "Priorizar una revisión individual de antigüedad y kilometraje antes de "
            "comparar vehículos: las medianas por segmento describen el conjunto, "
            "no el estado mecánico de cada unidad."
        ),
        (
            "Al ampliar el análisis, recolectar más anuncios de marcas actualmente "
            "menos representadas y conservar la moneda y procedencia para evaluar "
            "si los perfiles se mantienen."
        ),
    ]

    return {
        "dataset": {
            "archivo": "UsedCarInfo.csv",
            "registros": int(len(data)),
            "columnas_origen": SOURCE_COLUMNS,
            "marcas_distintas": int(data["brand"].nunique()),
            "fuente": "Mendeley Data, Md Saikot Hossain, DOI 10.17632/8d38h82fyt.2",
            "licencia": "CC BY 4.0",
        },
        "limpieza": audit,
        "variables_modelo": FEATURES,
        "transformaciones": [
            "Conversión de precio, kilometraje y capacidad de motor a valores numéricos.",
            "Extracción del valor seguido por cc para recuperar capacidad malformada.",
            "Transformación log1p para reducir la influencia de asimetrías y valores extremos.",
            "Estandarización StandardScaler después de aplicar log1p.",
            "No se eliminaron valores atípicos válidos detectados con la regla IQR.",
        ],
        "parametros": {
            "algoritmo": "KMeans",
            "random_state": RANDOM_STATE,
            "n_init": 20,
            "k_evaluados": [int(k) for k in K_VALUES],
            "silhouette_sample_size": min(SILHOUETTE_SAMPLE_SIZE, len(data)),
            "seleccion_k": "Mayor silhouette en la muestra fija; inercia se presenta como apoyo.",
        },
        "evaluacion_k": evaluation,
        "k_seleccionado": int(selected_k),
        "metricas_finales": next(
            row for row in evaluation if row["k"] == selected_k
        ),
        "comparacion_modelos": comparison,
        "perfiles_clusters": profile_rows,
        "centroides_retransformados": centroid_rows,
        "hallazgos": findings,
        "recomendaciones": recommendations,
        "limitaciones": [
            "Los anuncios describen un mercado y una fuente concretos; no representan todos los vehículos usados.",
            "La distribución de marcas está concentrada y puede sesgar la generalización.",
            "El precio es una de las variables de segmentación, por lo que los segmentos no son una predicción independiente del precio.",
            "K-Means favorece grupos aproximadamente compactos y depende de la escala, las variables y K elegidos.",
            "La silhouette se calcula en una muestra reproducible de hasta 2,000 filas para controlar el costo computacional.",
        ],
    }


def run_analysis(save_outputs: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ejecuta todo el flujo de preparación, comparación y K-Means."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el dataset original: {RAW_DATA_PATH}")

    original = pd.read_csv(RAW_DATA_PATH)
    data, audit = prepare_data(original)
    log_features = np.log1p(data[FEATURES].astype(float))
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(log_features)

    evaluation: list[dict[str, Any]] = []
    for k in K_VALUES:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(scaled_features)
        evaluation.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": float(
                    silhouette_score(
                        scaled_features,
                        labels,
                        sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(data)),
                        random_state=RANDOM_STATE,
                    )
                ),
                "davies_bouldin": float(
                    davies_bouldin_score(scaled_features, labels)
                ),
            }
        )
    selected_k = max(evaluation, key=lambda row: row["silhouette"])["k"]

    # fit_predict asigna un clúster a cada fila usando la configuración final.
    final_model = KMeans(
        n_clusters=selected_k, random_state=RANDOM_STATE, n_init=20
    )
    data["cluster"] = final_model.fit_predict(scaled_features)
    data["cluster"] = data["cluster"].astype(int)

    grouped = data.groupby("cluster", sort=True)
    cluster_profiles = grouped.agg(
        registros=("cluster", "size"),
        precio_mediano_bdt=("price_bdt", "median"),
        precio_promedio_bdt=("price_bdt", "mean"),
        antiguedad_mediana=("car_age", "median"),
        antiguedad_promedio=("car_age", "mean"),
        kilometraje_mediano=("kilometers_numeric", "median"),
        kilometraje_promedio=("kilometers_numeric", "mean"),
        motor_mediano_cc=("engine_cc", "median"),
        motor_promedio_cc=("engine_cc", "mean"),
    ).reset_index()
    cluster_profiles["porcentaje"] = (
        cluster_profiles["registros"] / len(data) * 100
    )

    global_price_quantiles = tuple(
        float(value) for value in data["price_bdt"].quantile([0.25, 0.75])
    )
    for row in cluster_profiles.itertuples(index=False):
        cluster_profiles.loc[
            cluster_profiles["cluster"] == row.cluster, "nombre"
        ] = _cluster_name(
            row.precio_mediano_bdt,
            row.antiguedad_mediana,
            row.kilometraje_mediano,
            global_price_quantiles,
            float(data["car_age"].median()),
            float(data["kilometers_numeric"].median()),
            int(row.cluster) + 1,
        )

    name_by_cluster = cluster_profiles.set_index("cluster")["nombre"].to_dict()
    data["cluster_name"] = data["cluster"].map(name_by_cluster)

    centroid_values = np.expm1(scaler.inverse_transform(final_model.cluster_centers_))
    centroids = pd.DataFrame(centroid_values, columns=FEATURES)
    centroids.insert(0, "cluster", np.arange(selected_k))
    centroids["nombre"] = centroids["cluster"].map(name_by_cluster)
    cluster_profiles["nombre"] = cluster_profiles["nombre"].astype(str)

    baseline_k = 3 if selected_k != 3 else 2
    selected_metrics = next(row for row in evaluation if row["k"] == selected_k)
    baseline_metrics = next(row for row in evaluation if row["k"] == baseline_k)
    baseline_model = KMeans(n_clusters=baseline_k, random_state=RANDOM_STATE, n_init=20)
    baseline_labels = baseline_model.fit_predict(scaled_features)
    baseline_counts = pd.Series(baseline_labels).value_counts().sort_index()
    comparison = {
        "modelo_seleccionado": {
            "k": int(selected_k),
            "silhouette": selected_metrics["silhouette"],
            "inertia": selected_metrics["inertia"],
            "davies_bouldin": selected_metrics["davies_bouldin"],
            "tamanos": [int(value) for value in data["cluster"].value_counts().sort_index()],
        },
        "modelo_referencia": {
            "k": int(baseline_k),
            "silhouette": baseline_metrics["silhouette"],
            "inertia": baseline_metrics["inertia"],
            "davies_bouldin": baseline_metrics["davies_bouldin"],
            "tamanos": [int(value) for value in baseline_counts],
        },
    }
    silhouette_difference = (
        comparison["modelo_seleccionado"]["silhouette"]
        - comparison["modelo_referencia"]["silhouette"]
    )
    comparison["interpretacion"] = (
        f"K={selected_k} obtuvo silhouette "
        f"{comparison['modelo_seleccionado']['silhouette']:.4f} frente a "
        f"{comparison['modelo_referencia']['silhouette']:.4f} con K={baseline_k} "
        f"(diferencia {silhouette_difference:+.4f}). La inercia pasó de "
        f"{comparison['modelo_referencia']['inertia']:,.2f} a "
        f"{comparison['modelo_seleccionado']['inertia']:,.2f} y los tamaños de grupo "
        f"fueron {comparison['modelo_referencia']['tamanos']} frente a "
        f"{comparison['modelo_seleccionado']['tamanos']}. K seleccionado maximiza "
        "silhouette entre los valores probados; la distribución desigual de tamaños "
        "debe tenerse en cuenta al usar los perfiles."
    )
    summary = _build_summary(
        data,
        audit,
        evaluation,
        int(selected_k),
        cluster_profiles,
        centroids,
        comparison,
    )

    if save_outputs:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8-sig")
        RESULTS_PATH.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _make_figures(data, evaluation, int(selected_k), centroids)

    return data, summary


if __name__ == "__main__":
    analyzed_data, results = run_analysis()
    print(f"Registros procesados: {len(analyzed_data):,}")
    print(f"K seleccionado: {results['k_seleccionado']}")
    print(f"Silhouette: {results['metricas_finales']['silhouette']:.4f}")
    print(f"CSV procesado: {PROCESSED_DATA_PATH.relative_to(ROOT_DIR)}")
    print(f"Resultados: {RESULTS_PATH.relative_to(ROOT_DIR)}")
