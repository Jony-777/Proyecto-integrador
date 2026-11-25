"""Exploración inicial del dataset original (Parte 2 del proyecto).

Muestra los primeros 10 registros, dimensiones, columnas, tipos de datos,
estadísticas descriptivas, valores nulos y duplicados. El resultado se imprime
en pantalla y se guarda en reports/exploracion_dataset.txt.

Ejecución:  python -m src.exploracion
"""

from __future__ import annotations

import io

import pandas as pd

from src.analysis import FEATURES, RAW_DATA_PATH, ROOT_DIR, prepare_data

OUTPUT_PATH = ROOT_DIR / "reports" / "exploracion_dataset.txt"


def build_exploration_report() -> str:
    """Genera el texto de la exploración a partir del CSV original."""
    original = pd.read_csv(RAW_DATA_PATH)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.float_format", "{:,.2f}".format)

    out = io.StringIO()

    def section(title: str) -> None:
        out.write("\n" + "=" * 80 + "\n" + title + "\n" + "=" * 80 + "\n")

    section("1. Primeros 10 registros")
    out.write(original.head(10).to_string() + "\n")

    section("2. Dimensiones del dataset")
    rows, cols = original.shape
    out.write(f"Registros (filas): {rows:,}\nVariables (columnas): {cols}\n")

    section("3. Columnas y tipos de datos")
    types = pd.DataFrame(
        {
            "tipo_de_dato": original.dtypes.astype(str),
            "valores_unicos": original.nunique(),
        }
    )
    out.write(types.to_string() + "\n")
    out.write(
        "\nNota: price, energy_capacity y kilometers_run son texto con unidades "
        "('Tk 530,000', '1,500 cc', '89,000 km'); se convierten a número en "
        "src/analysis.py (prepare_data).\n"
    )

    section("4. Valores nulos por columna")
    out.write(original.isna().sum().to_string() + "\n")

    section("5. Registros duplicados")
    out.write(f"Duplicados exactos: {int(original.duplicated().sum())}\n")

    section("6. Estadísticas descriptivas (columnas numéricas originales)")
    out.write(original.describe().round(2).to_string() + "\n")

    section("7. Estadísticas descriptivas de las variables del modelo (ya convertidas)")
    prepared, _ = prepare_data(original)
    labels = {
        "price_bdt": "precio_bdt",
        "car_age": "antiguedad_anios",
        "kilometers_numeric": "kilometraje_km",
        "engine_cc": "motor_cc",
    }
    out.write(
        prepared[FEATURES].rename(columns=labels).describe().round(2).to_string() + "\n"
    )

    section("8. Variables categóricas principales")
    out.write("Marcas con más anuncios (top 5, % del total):\n")
    out.write(
        (original["brand"].value_counts(normalize=True).head(5) * 100).round(2).to_string()
        + "\n\n"
    )
    out.write("Transmisión:\n" + original["transmission"].value_counts().to_string() + "\n")

    section("9. Variables seleccionadas para el agrupamiento y justificación")
    out.write(
        "- price (precio_bdt): refleja el valor de mercado del vehículo.\n"
        "- car_age (antigüedad): indica qué tan nuevo es el vehículo; se usa en lugar\n"
        "  de year_of_manufacture para no duplicar la misma información.\n"
        "- kilometers_run (kilometraje): mide el uso o desgaste del vehículo.\n"
        "- energy_capacity (motor_cc): distingue vehículos compactos de los más grandes.\n"
        "Son numéricas, comparables entre marcas y describen valor y desgaste.\n"
        "Las variables categóricas (marca, transmisión, combustible) se usan como\n"
        "filtros del dashboard.\n"
    )
    return out.getvalue()


if __name__ == "__main__":
    report = build_exploration_report()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Exploración guardada en: {OUTPUT_PATH.relative_to(ROOT_DIR)}")
