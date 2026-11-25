"""Dashboard interactivo para explorar anuncios y segmentos de vehículos."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import run_analysis

st.set_page_config(
    page_title="Segmentación de vehículos usados",
    page_icon="🚘",
    layout="wide",
)


@st.cache_data(show_spinner="Preparando datos y segmentos...")
def load_analysis() -> tuple[pd.DataFrame, dict]:
    """Carga el resultado reproducible del análisis sobre el CSV original."""
    return run_analysis()


def format_bdt(value: float) -> str:
    """Presenta importes en taka de Bangladesh sin inventar conversiones."""
    return f"Tk {value:,.0f}"


def format_bdt_compact(value: float) -> str:
    """Acorta importes para que los KPI sean legibles en una columna."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,.0f}"


try:
    cars, results = load_analysis()
except (FileNotFoundError, ValueError) as error:
    st.error(f"No se pudo cargar el análisis: {error}")
    st.stop()

st.title("Segmentación de vehículos usados con K-Means")
st.caption(
    "Explora anuncios del mercado de Bangladesh. Los perfiles son descriptivos y "
    "no constituyen una tasación."
)

with st.sidebar:
    st.header("Filtros")
    brands = sorted(cars["brand"].dropna().unique().tolist())
    transmissions = sorted(cars["transmission"].dropna().unique().tolist())
    fuel_types = sorted(cars["fuel_type"].dropna().unique().tolist())
    selected_brands = st.multiselect("Marca", brands, default=brands)
    selected_transmissions = st.multiselect(
        "Transmisión", transmissions, default=transmissions
    )
    selected_fuel_types = st.multiselect(
        "Tipo de combustible", fuel_types, default=fuel_types
    )

filtered = cars[
    cars["brand"].isin(selected_brands)
    & cars["transmission"].isin(selected_transmissions)
    & cars["fuel_type"].isin(selected_fuel_types)
].copy()
filtered["cluster_code"] = filtered["cluster"].map(
    lambda cluster: f"C{cluster + 1}"
)

if filtered.empty:
    st.info("No hay anuncios que coincidan con los tres filtros seleccionados.")
    st.stop()

general_tab, segments_tab = st.tabs(["1. Análisis general", "2. Segmentación K-Means"])

with general_tab:
    st.subheader("Panorama de los anuncios filtrados")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Anuncios", f"{len(filtered):,}")
    median_price = float(filtered["price_bdt"].median())
    metric_columns[1].metric(
        "Precio mediano",
        f"{format_bdt_compact(median_price)} Tk",
        help=f"Valor exacto: {format_bdt(median_price)}",
    )
    metric_columns[2].metric(
        "Antigüedad mediana", f"{filtered['car_age'].median():.1f} años"
    )
    median_mileage = float(filtered["kilometers_numeric"].median())
    metric_columns[3].metric(
        "Km mediana",
        f"{median_mileage / 1_000:.1f}k km"
        if median_mileage >= 1_000
        else f"{median_mileage:.0f} km",
        help=f"Valor exacto: {median_mileage:,.0f} km",
    )

    price_chart, brand_chart = st.columns(2)
    with price_chart:
        figure = px.histogram(
            filtered,
            x="price_bdt",
            nbins=35,
            log_x=True,
            title="Distribución de precios (eje logarítmico)",
            labels={"price_bdt": "Precio (BDT)", "count": "Anuncios"},
        )
        st.plotly_chart(figure, width="stretch")
    with brand_chart:
        brand_counts = (
            filtered["brand"].value_counts().head(10).rename_axis("Marca").reset_index(name="Anuncios")
        )
        figure = px.bar(
            brand_counts,
            x="Anuncios",
            y="Marca",
            orientation="h",
            title="Marcas con más anuncios en el filtro",
        )
        figure.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(figure, width="stretch")

    age_chart, transmission_chart = st.columns(2)
    with age_chart:
        figure = px.histogram(
            filtered,
            x="car_age",
            nbins=25,
            title="Antigüedad declarada",
            labels={"car_age": "Antigüedad (años)", "count": "Anuncios"},
        )
        st.plotly_chart(figure, width="stretch")
    with transmission_chart:
        transmission_counts = (
            filtered["transmission"]
            .value_counts()
            .rename_axis("Transmisión")
            .reset_index(name="Anuncios")
        )
        figure = px.pie(
            transmission_counts,
            values="Anuncios",
            names="Transmisión",
            title="Anuncios por transmisión",
        )
        st.plotly_chart(figure, width="stretch")

    with st.expander("Vista previa y estadísticas del dataset"):
        st.markdown("**Primeros 10 registros (columnas originales)**")
        st.dataframe(
            filtered[
                [
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
            ].head(10),
            width="stretch",
            hide_index=True,
        )
        st.markdown("**Estadísticas descriptivas de las variables del modelo**")
        st.dataframe(
            filtered[["price_bdt", "car_age", "kilometers_numeric", "engine_cc"]]
            .rename(
                columns={
                    "price_bdt": "Precio (BDT)",
                    "car_age": "Antigüedad (años)",
                    "kilometers_numeric": "Kilometraje (km)",
                    "engine_cc": "Motor (cc)",
                }
            )
            .describe()
            .style.format("{:,.2f}"),
            width="stretch",
        )

    st.caption(
        f"Fuente indicada para el dataset: {results['dataset']['fuente']}. "
        f"Licencia declarada: {results['dataset']['licencia']}."
    )

with segments_tab:
    st.subheader(f"Perfiles K-Means (K = {results['k_seleccionado']})")
    cluster_counts = (
        filtered.groupby(["cluster", "cluster_code"], as_index=False)
        .size()
        .rename(columns={"size": "Anuncios"})
        .sort_values("cluster")
    )
    count_chart, scatter_chart = st.columns(2)
    with count_chart:
        figure = px.bar(
            cluster_counts,
            x="cluster_code",
            y="Anuncios",
            color="cluster_code",
            title="Distribución de anuncios por clúster",
            labels={"cluster_code": "Segmento"},
        )
        figure.update_layout(showlegend=False, xaxis_tickangle=-25)
        st.plotly_chart(figure, width="stretch")
    with scatter_chart:
        figure = px.scatter(
            filtered,
            x="car_age",
            y="price_bdt",
            color="cluster_code",
            hover_data=[
                "cluster_name",
                "brand",
                "model",
                "kilometers_numeric",
                "engine_cc",
                "transmission",
                "fuel_type",
            ],
            log_y=True,
            title="Antigüedad y precio por segmento",
            labels={
                "car_age": "Antigüedad (años)",
                "price_bdt": "Precio (BDT, escala logarítmica)",
                "cluster_code": "Segmento",
            },
            opacity=0.7,
        )
        st.plotly_chart(figure, width="stretch")

    price_chart, mileage_chart = st.columns(2)
    with price_chart:
        figure = px.box(
            filtered,
            x="cluster_code",
            y="price_bdt",
            color="cluster_code",
            title="Comparación de precios por segmento",
            labels={"cluster_code": "Segmento", "price_bdt": "Precio (BDT)"},
            points=False,
        )
        figure.update_yaxes(type="log")
        figure.update_layout(showlegend=False, xaxis_tickangle=-25)
        st.plotly_chart(figure, width="stretch")
    with mileage_chart:
        filtered_profiles = (
            filtered.groupby("cluster_code", as_index=False)
            .agg(
                Kilometraje_mediano=("kilometers_numeric", "median"),
                Motor_mediano_cc=("engine_cc", "median"),
            )
        )
        figure = px.bar(
            filtered_profiles,
            x="cluster_code",
            y="Kilometraje_mediano",
            color="cluster_code",
            title="Kilometraje mediano por segmento",
            labels={
                "cluster_code": "Segmento",
                "Kilometraje_mediano": "Kilometraje mediano (km)",
            },
        )
        figure.update_layout(showlegend=False, xaxis_tickangle=-25)
        st.plotly_chart(figure, width="stretch")

    st.markdown("#### Estadísticas descriptivas por clúster")
    profiles = (
        filtered.groupby(["cluster", "cluster_name"], as_index=False)
        .agg(
            registros=("cluster", "size"),
            precio_mediano_bdt=("price_bdt", "median"),
            precio_promedio_bdt=("price_bdt", "mean"),
            antiguedad_mediana=("car_age", "median"),
            antiguedad_promedio=("car_age", "mean"),
            kilometraje_mediano=("kilometers_numeric", "median"),
            kilometraje_promedio=("kilometers_numeric", "mean"),
            motor_mediano_cc=("engine_cc", "median"),
            motor_promedio_cc=("engine_cc", "mean"),
        )
        .sort_values("cluster")
    )
    profiles["porcentaje"] = profiles["registros"] / len(filtered) * 100
    profile_descriptions = profiles.to_dict(orient="records")
    profiles = profiles[
        [
            "cluster",
            "cluster_name",
            "registros",
            "porcentaje",
            "precio_mediano_bdt",
            "precio_promedio_bdt",
            "antiguedad_mediana",
            "antiguedad_promedio",
            "kilometraje_mediano",
            "kilometraje_promedio",
            "motor_mediano_cc",
            "motor_promedio_cc",
        ]
    ].rename(
        columns={
            "cluster": "ID",
            "cluster_name": "Segmento",
            "registros": "Registros",
            "porcentaje": "Participación (%)",
            "precio_mediano_bdt": "Precio mediano (BDT)",
            "precio_promedio_bdt": "Precio promedio (BDT)",
            "antiguedad_mediana": "Edad mediana (años)",
            "antiguedad_promedio": "Edad promedio (años)",
            "kilometraje_mediano": "Km mediana",
            "kilometraje_promedio": "Km promedio",
            "motor_mediano_cc": "Motor mediano (cc)",
            "motor_promedio_cc": "Motor promedio (cc)",
        }
    )
    st.dataframe(
        profiles.style.format(
            {
                "Participación (%)": "{:.1f}",
                "Precio mediano (BDT)": "{:,.0f}",
                "Precio promedio (BDT)": "{:,.0f}",
                "Edad mediana (años)": "{:.1f}",
                "Edad promedio (años)": "{:.1f}",
                "Km mediana": "{:,.1f}",
                "Km promedio": "{:,.0f}",
                "Motor mediano (cc)": "{:,.0f}",
                "Motor promedio (cc)": "{:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Centroides del modelo K-Means")
    centroids = pd.DataFrame(results["centroides_retransformados"])
    centroids["Segmento"] = centroids["cluster"].map(lambda value: f"C{value + 1}")
    centroids = centroids[
        ["Segmento", "nombre", "price_bdt", "car_age", "kilometers_numeric", "engine_cc"]
    ].rename(
        columns={
            "nombre": "Descripción",
            "price_bdt": "Precio (BDT)",
            "car_age": "Antigüedad (años)",
            "kilometers_numeric": "Kilometraje (km)",
            "engine_cc": "Motor (cc)",
        }
    )
    st.dataframe(
        centroids.style.format(
            {
                "Precio (BDT)": "{:,.0f}",
                "Antigüedad (años)": "{:.1f}",
                "Kilometraje (km)": "{:,.0f}",
                "Motor (cc)": "{:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    centroid_values = pd.DataFrame(results["centroides_retransformados"])
    centroid_values.index = centroid_values["cluster"].map(lambda value: f"C{value + 1}")
    feature_labels = {
        "price_bdt": "Precio (BDT)",
        "car_age": "Antigüedad (años)",
        "kilometers_numeric": "Kilometraje (km)",
        "engine_cc": "Motor (cc)",
    }
    centroid_values = centroid_values[list(feature_labels)]
    relative_centroids = (
        centroid_values / cars[list(feature_labels)].median() * 100
    ).rename(columns=feature_labels)
    centroid_text = centroid_values.apply(
        lambda column: column.map(
            (lambda value: f"{value:.1f}")
            if column.name == "car_age"
            else (lambda value: f"{value:,.0f}")
        )
    )
    figure = px.imshow(
        relative_centroids.clip(upper=300),
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=100,
        zmin=0,
        zmax=300,
        aspect="auto",
        labels={"x": "Variable", "y": "Segmento", "color": "% de la mediana"},
        title="Centroides comparados con la mediana global (texto = valor real)",
    )
    figure.update_traces(text=centroid_text.to_numpy(), texttemplate="%{text}")
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Centroides calculados sobre el dataset completo y regresados a unidades "
        "originales (inversa de StandardScaler y log1p). No cambian con los filtros."
    )

    st.markdown("#### Descripción de los grupos")
    for profile in profile_descriptions:
        st.markdown(
            f"- **C{profile['cluster'] + 1} — {profile['cluster_name']}**: "
            f"{profile['registros']:,} vehículos ({profile['porcentaje']:.1f}%); "
            f"mediana de precio {format_bdt(profile['precio_mediano_bdt'])}, "
            f"antigüedad {profile['antiguedad_mediana']:.1f} años y "
            f"kilometraje {profile['kilometraje_mediano']:,.1f} km."
        )

    st.markdown("#### Hallazgos calculados sobre el dataset completo")
    for finding in results["hallazgos"]:
        st.markdown(f"- {finding}")

    st.markdown("#### Recomendaciones")
    for recommendation in results["recomendaciones"]:
        st.markdown(f"- {recommendation}")

    st.markdown("#### Evaluación del modelo")
    metrics = results["metricas_finales"]
    comparison = results["comparacion_modelos"]
    st.write(
        f"K elegido por silhouette máximo entre los valores evaluados: "
        f"**{metrics['k']}**. Silhouette (muestra): "
        f"**{metrics['silhouette']:.4f}**; inercia: "
        f"**{metrics['inertia']:,.2f}**; Davies–Bouldin: "
        f"**{metrics['davies_bouldin']:.4f}**."
    )
    st.write(comparison["interpretacion"])
    st.caption(
        "Las métricas comparan soluciones sobre variables transformadas con log1p "
        "y estandarizadas. Silhouette se estimó con una muestra reproducible."
    )
