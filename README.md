# Segmentación de vehículos usados con K-Means

Proyecto en equipo de minería de datos para describir segmentos de
anuncios de vehículos usados y explorarlos en un dashboard de Streamlit.

## Problema y objetivos

Los anuncios presentan diferencias de precio, antigüedad, kilometraje y capacidad
del motor. Se busca explorar qué perfiles aparecen al analizar conjuntamente esas
características, sin tratar los grupos como una predicción de precio o una
tasación.

**Justificación:** revisar miles de anuncios uno por uno no permite saber qué
tipos de vehículos se ofrecen ni cuántos hay de cada tipo. Agrupar los anuncios
con K-Means resume el mercado en pocos segmentos interpretables, útiles para
compradores (comparar un vehículo con otros similares), vendedores (posicionar
su oferta) y analistas (conocer la composición del inventario). Se espera
encontrar grupos como vehículos antiguos de bajo precio y alto recorrido frente
a vehículos recientes, de poco recorrido y precio alto.

**Objetivo general:** aplicar K-Means para identificar y describir grupos de
vehículos usados y presentar los resultados en un dashboard interactivo.

**Objetivos específicos:**

1. Explorar la calidad, distribución y tipos de datos del CSV.
2. Limpiar y transformar variables numéricas, evaluar cinco valores de K y
   justificar el modelo final mediante métricas.
3. Interpretar centroides y perfiles, y comunicar los resultados y limitaciones
   con visualizaciones y filtros.

## Dataset y atribución

- Archivo original: `UsedCarInfo.csv` (5,997 filas y 10 columnas).
- Fuente proporcionada para este proyecto: Mendeley Data, Md Saikot Hossain,
  [DOI 10.17632/8d38h82fyt.2](https://doi.org/10.17632/8d38h82fyt.2).
- Licencia declarada: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- El conjunto describe anuncios de Bangladesh y el precio está en taka
  bangladesí (BDT). No representa necesariamente todos los mercados ni todos los
  vehículos usados.
- `data_dictionary.xlsx` y `collection_methodology.pdf` se conservan como
  materiales de referencia distribuidos con los archivos del dataset.

Las columnas originales del CSV son `title`, `brand`, `model`, `transmission`,
`fuel_type`, `price`, `year_of_manufacture`, `energy_capacity`,
`kilometers_run` y `car_age`. Los valores de precio, capacidad y kilometraje se
almacenan como texto con unidades o separadores; el procesamiento los convierte
a números sin modificar el original.

## Exploración del dataset

`python -m src.exploracion` muestra los primeros 10 registros, dimensiones,
columnas, tipos de datos, valores nulos, duplicados y estadísticas descriptivas,
y guarda el resultado en `reports/exploracion_dataset.txt`. Las gráficas
exploratorias (precios, antigüedad y marcas) se generan en `reports/figures/`.

**Variables seleccionadas:** precio, antigüedad, kilometraje y capacidad del
motor. Son numéricas, comparables entre marcas y describen el valor y el
desgaste del vehículo. Se usa `car_age` en lugar de `year_of_manufacture` para
no duplicar la misma información; las variables categóricas (marca, transmisión
y combustible) se usan como filtros del dashboard.

## Preparación y modelado

Herramientas: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn,
Streamlit y Plotly.

Se usan cuatro variables para el agrupamiento: precio BDT, antigüedad,
kilometraje y capacidad del motor. El campo `energy_capacity` contiene una fila
con texto incrustado; se recupera el número asociado a `cc` y se conserva la
fila. Los duplicados exactos se eliminan si existen; no se imputan nulos porque
el CSV inspeccionado no contiene valores nulos. Los valores extremos se
identifican con IQR y se conservan por poder representar anuncios válidos.

Para reducir la influencia de asimetrías se aplica `log1p` y luego
`StandardScaler`. Se comparan K=2, 3, 4, 5 y 6 con K-Means, semilla 42 y
`n_init=20`; se selecciona el valor con mayor silhouette en una muestra
reproducible de hasta 2,000 filas. La inercia y Davies-Bouldin también se
reportan como métricas de apoyo. Los resultados exactos y los hallazgos se
generan desde el CSV al ejecutar el flujo, no se escriben como cifras
preestablecidas.

### Resultados del CSV inspeccionado

La ejecución sobre el archivo incluido seleccionó **K=3**: silhouette **0.4624**
(muestra de 2,000), Davies-Bouldin **0.7722** e inercia **11,899.88**. Frente a
la referencia K=2 (silhouette **0.4219**, inercia **16,747.12**), K=3 obtuvo un
silhouette mayor y formó tres grupos de **1,152**, **4,693** y **152** anuncios.
El grupo más pequeño representa **2.5%** del conjunto, por lo que su perfil
debe interpretarse con cautela.

| Grupo descriptivo | Anuncios | Mediana de precio (BDT) | Mediana de edad | Mediana de kilometraje |
|---|---:|---:|---:|---:|
| Precio bajo, mayor antigüedad y mayor recorrido (C1) | 1,152 | 1,395,000 | 18 años | 78,000 km |
| Precio intermedio, más recientes y menor recorrido (C2) | 4,693 | 3,640,000 | 6 años | 58,748 km |
| Precio alto, más recientes y menor recorrido (C3) | 152 | 6,300,000 | 1.5 años | 14.5 km |

El CSV real tenía **5,997 filas**, **10 columnas**, ningún valor nulo ni
duplicado exacto. Se recuperó la capacidad de motor en una fila con texto
incrustado. Los valores atípicos detectados no se eliminaron. Todos los
hallazgos de arriba se recalculan al ejecutar el análisis.

## Ejecución

Requiere Python 3.10 o posterior.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.exploracion
python -m src.analysis
python -m src.generate_report
streamlit run app.py
```

El análisis genera `data/used_cars_clustered.csv`,
`reports/analysis_results.json` y gráficas en `reports/figures/`: tres
exploratorias (`distribucion_precios`, `distribucion_antiguedad`,
`anuncios_por_marca`), la evaluación de K (`evaluacion_k`), dos dispersiones por
clúster (`segmentacion_kmeans`, `segmentacion_antiguedad_precio`), la comparación
de variables por clúster (`comparacion_clusters`) y el mapa de centroides
(`centroides_kmeans`). El informe
PDF se guarda en `reports/informe_segmentacion_vehiculos.pdf`. Streamlit
presenta dos secciones: análisis general (KPI, gráficas generales y vista previa
con estadísticas del dataset) y segmentación K-Means (distribución por clúster,
dispersión, comparaciones, estadísticas por grupo, tabla y gráfica de centroides, descripción de
grupos, hallazgos, recomendaciones y métricas del modelo). Sus filtros de
marca, transmisión y combustible actualizan las métricas y visualizaciones.

## Documentación

La documentación del proyecto (Parte 7) está en
`docs/Documentacion_Proyecto_KMeans_BBJ.pdf`. El archivo
`reports/informe_segmentacion_vehiculos.pdf` es un informe técnico complementario
que el código genera automáticamente.

Para ejecutar las pruebas:

```powershell
python -m unittest discover -s tests -v
```

## Estructura

```text
.
├── UsedCarInfo.csv
├── data_dictionary.xlsx
├── collection_methodology.pdf
├── app.py
├── docs/
│   └── Documentacion_Proyecto_KMeans_BBJ.pdf  # documentación del proyecto
├── data/
│   └── used_cars_clustered.csv       # generado
├── reports/
│   ├── analysis_results.json         # generado
│   ├── exploracion_dataset.txt       # generado (exploración inicial)
│   ├── informe_segmentacion_vehiculos.pdf  # generado
│   └── figures/                      # gráficas generadas
├── src/
│   ├── exploracion.py
│   ├── analysis.py
│   └── generate_report.py
├── tests/
│   └── test_analysis.py
└── requirements.txt
```

## Referencia descriptiva del dataset de origen

El archivo de datos contiene anuncios recopilados de mercados automotrices de
Bangladesh. El conjunto está fuertemente representado por Toyota; `fuel_type`
puede incluir combinaciones de combustibles. `title` es el título de la
publicación y no se usa como variable directa del modelo. `year_of_manufacture`
se conserva para contexto y verificación; la segmentación usa la edad declarada
para evitar duplicar medidas de antigüedad.
