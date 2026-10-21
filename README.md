# Proyecto de clustering: CLEF gambling

El proyecto incluye preprocesamiento, representaciones TF-IDF/SVD y clustering
jerárquico. Las representaciones están en `resultados_tfidf/` en dos conjuntos:
`datos_1_a_600` (ocho representaciones de 598 usuarios) y `datos_completos`
(seis representaciones de 2.344 usuarios, las cinco carpetas de CLEF). Ver
[resultados_tfidf/README.md](resultados_tfidf/README.md). Los datos XML y los
CSV del preprocesamiento se mantienen fuera de GitHub.

## Empezar con los experimentos

Desde la raíz de `MD`, con Python 3.10 o superior:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r experimentos/requirements.txt
.\.venv\Scripts\python.exe -m experimentos.todas --comprobar-entradas
.\.venv\Scripts\python.exe -m experimentos.todas --estabilidad 5
.\.venv\Scripts\python.exe -m experimentos.todas --conjunto datos_completos --estabilidad 5
```

La [guía de las ocho representaciones](experimentos/GUIA_TODAS.md) explica las
carpetas, los históricos alineados y cómo interpretar la comparación.

## Clustering y búsqueda de k

La interfaz de clustering admite matrices dispersas y densas, Ward y las
métricas silhouette, Davies-Bouldin y Calinski-Harabasz:

```powershell
.\.venv\Scripts\python.exe -m pip install -r Clustering_jerarquico/requirements.txt
.\.venv\Scripts\python.exe -m Clustering_jerarquico --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo_svd95pct --salida resultados_clustering_svd95pct
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

## La cadena completa, desde los XML

Si partes de cero y quieres rehacerlo todo, este es el orden. Desde la raíz de
`MD`, y con los XML de CLEF en una carpeta local:

```powershell
python -m preprocesador --datos "<ruta a los XML>" --guardar_csv
python -m vectorizacion --salida resultados_prueba --quitar-vacias --sublinear-tf
python -m vectorizacion.svd --entrada resultados_prueba --dimensiones 100
python -m Clustering_jerarquico experimentar --entrada resultados_prueba_svd100
python -m experimentos --entrada resultados_prueba_svd100
python -m Clustering_jerarquico modelo --entrada resultados_prueba_svd100
```

El preprocesamiento tarda un par de minutos; el resto, segundos. Cada paso deja
su carpeta de resultados y el siguiente la lee.

## Consultar un texto nuevo

Con el modelo ya entrenado, asigna un texto a un cluster y enseña los usuarios
más parecidos dentro de él:

```powershell
python -m experimentos.consultar "I cannot stop gambling" --entrada resultados_prueba_svd100 --grafico consulta.png
```

El gráfico sitúa a cada usuario por su **distancia real** al texto, sin proyectar
a 2D: si los vecinos aparecen pegados al cero, el texto ha encontrado a su gente;
si salen amontonados con todos los demás, ha caído en ese cluster por descarte.

Los detalles están en el [README del banco de pruebas](experimentos/README.md).

## Pruebas de integración

Instala también las dependencias de `vectorizacion/requirements.txt` y ejecuta:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s vectorizacion -t .
.\.venv\Scripts\python.exe -m unittest discover -s experimentos -t .
.\.venv\Scripts\python.exe -m unittest discover -s Clustering_jerarquico -t .
```
