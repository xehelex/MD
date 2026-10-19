# Proyecto de clustering: CLEF gambling

El proyecto incluye preprocesamiento, ocho representaciones comparables de los
mismos 598 usuarios y clustering jerárquico. Los datos XML y los CSV del
preprocesamiento se mantienen fuera de GitHub.

## Empezar con los experimentos

Desde la raíz de `MD`, con Python 3.10 o superior:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r experimentos/requirements.txt
.\.venv\Scripts\python.exe -m experimentos.todas --comprobar-entradas
.\.venv\Scripts\python.exe -m experimentos.todas --estabilidad 5
```

La [guía de las ocho representaciones](experimentos/GUIA_TODAS.md) explica las
carpetas, los históricos alineados y cómo interpretar la comparación.

## Clustering y búsqueda de k

La interfaz de clustering admite matrices dispersas y densas, Ward y las
métricas silhouette, Davies-Bouldin y Calinski-Harabasz:

```powershell
.\.venv\Scripts\python.exe -m pip install -r Clustering_jerarquico/requirements.txt
.\.venv\Scripts\python.exe -m Clustering_jerarquico --entrada resultados_tfidf_completo_svd95pct --salida resultados_clustering_svd95pct
```

Consulta la [documentación del clustering](Clustering_jerarquico/README.md) para
configurar enlaces, gráficas y experimentos posteriores.

## Regenerar o transformar usuarios nuevos

La [documentación de vectorización](vectorizacion/README.md) incluye TF-IDF,
SVD, el análisis de varianza y la preparación de los históricos. Para cargar
los modelos SVD guardados, instala Git LFS y ejecuta:

```powershell
git lfs install
git lfs pull
```

Los cuatro modelos SVD están en LFS por su tamaño. Las matrices y los IDs están
en Git normal y bastan para experimentar con los usuarios existentes.

## Pruebas de integración

Instala también las dependencias de `vectorizacion/requirements.txt` y ejecuta:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s vectorizacion -t .
.\.venv\Scripts\python.exe -m unittest discover -s experimentos -t .
.\.venv\Scripts\python.exe -m unittest discover -s Clustering_jerarquico -t .
```
