# Experimentos con las ocho representaciones

Ejecuta desde la raíz de `MD`, con Python 3.10 o superior. Para experimentar con
las matrices ya generadas no necesitas los XML, los CSV del preprocesado ni los
modelos `.joblib`: necesitas el código, `X.npz` o `X.npy` y `usuarios.json` de
cada una de estas ocho carpetas. Conserva también los resúmenes para interpretar
los resultados.

| Carpeta | Dimensiones | Origen |
|---|---:|---|
| `resultados_tfidf_historico_alineado` | 37.246 | TF-IDF histórico de palabras, train y test unidos. |
| `resultados_tfidf_caracteres4_historico_alineado` | 138.112 | TF-IDF histórico de caracteres, train y test unidos. |
| `resultados_tfidf_completo` | 29.068 | TF-IDF de palabras sobre el preprocesado actual. |
| `resultados_tfidf_caracteres4_completo` | 67.933 | TF-IDF de caracteres sobre el preprocesado actual. |
| `resultados_tfidf_completo_svd100` | 100 | SVD de palabras. |
| `resultados_tfidf_caracteres4_completo_svd100` | 100 | SVD de caracteres. |
| `resultados_tfidf_completo_svd95pct` | 495 | SVD de palabras: 95,04 % de varianza. |
| `resultados_tfidf_caracteres4_completo_svd95pct` | 479 | SVD de caracteres: 95,04 % de varianza. |

Todas tienen los mismos **598 usuarios en el mismo orden**. El banco de pruebas
rechaza la comparación si las listas de usuarios no coinciden.

## Preparar el entorno y comprobar archivos

Actualiza el repositorio con `git pull --ff-only`. Los modelos SVD `.joblib` se
distribuyen con Git LFS. Para utilizarlos al transformar usuarios nuevos, instala
Git LFS y ejecuta `git lfs install` y `git lfs pull` desde `MD`.
Las matrices y los listados de usuarios están en Git normal: los experimentos
sobre usuarios ya vectorizados no necesitan descargar los modelos LFS.

Si aún no tienes un entorno, créalo con `python -m venv .venv`. Después:

```powershell
.\.venv\Scripts\python.exe -m pip install -r experimentos/requirements.txt
.\.venv\Scripts\python.exe -m experimentos.todas --comprobar-entradas
```

El segundo comando comprueba que las ocho matrices y sus IDs son válidos. No
ejecuta clustering ni guarda resultados.

## Ejecutar la comparación

```powershell
.\.venv\Scripts\python.exe -m experimentos.todas
```

Evalúa las cinco combinaciones de distancia/enlace del banco existente, para
valores de k entre 2 y 20. Guarda `rejilla.csv` y `resumen.json` en
`resultados_experimentos_todas`. Para repetir, elige otro nombre con `--salida`
si quieres conservar el informe anterior.

Para incluir la estabilidad con cinco repeticiones:

```powershell
.\.venv\Scripts\python.exe -m experimentos.todas --estabilidad 5 --salida resultados_experimentos_todas_estabilidad
```

El lanzador admite las mismas opciones que `python -m experimentos`, como
`--k-min`, `--k-max`, `--minimo` y `--salida`. El remuestreo toma el 80 % de
usuarios sin reemplazo y mantiene los vectores fijos: no vuelve a ajustar
TF-IDF ni SVD. Valora conjuntamente silhouette, equilibrio, tamaños y estabilidad.

La interfaz de `Clustering_jerarquico` de los compañeros también acepta todas
estas carpetas y conserva sus gráficas, Ward, Davies-Bouldin y Calinski-Harabasz:

```powershell
.\.venv\Scripts\python.exe -m pip install -r Clustering_jerarquico/requirements.txt
.\.venv\Scripts\python.exe -m Clustering_jerarquico --entrada resultados_tfidf_completo_svd95pct --salida resultados_clustering_svd95pct
```

Su configuración controla los enlaces, el rango de k y los experimentos
posteriores. Estos utilizan la misma entrada y salida seleccionadas en el comando.

## Interpretar los históricos

Las carpetas originales `resultados_tfidf` y `resultados_tfidf_caracteres4`
siguen intactas con train/test. Para los experimentos se usan sus nuevas copias
`*_historico_alineado`: se unieron las 480 filas de train y 120 de test, se
excluyeron `subject2538` y `subject2569`, y se ordenaron las 598 filas restantes
como las representaciones actuales. No se recalculó ningún valor TF-IDF.

Los históricos conservan su procesamiento anterior y el vocabulario/IDF
aprendidos con **480 usuarios**. Los actuales se ajustaron con **598 usuarios**
del nuevo preprocesado. Compararlos evalúa ambas variantes completas; una
diferencia de resultados no puede atribuirse únicamente al preprocesamiento.
La comparación entre un TF-IDF actual y sus variantes SVD sí mantiene el mismo
TF-IDF de origen y permite estudiar la reducción de dimensiones.

## Regenerar las copias históricas alineadas

Conserva las dos carpetas históricas originales y `resultados_tfidf_completo`
como referencia. Instala las dependencias de `vectorizacion/requirements.txt` y ejecuta:

```powershell
.\.venv\Scripts\python.exe -m vectorizacion.preparar_historicos --entrada resultados_tfidf
.\.venv\Scripts\python.exe -m vectorizacion.preparar_historicos --entrada resultados_tfidf_caracteres4
```

Si las salidas ya existen, indica carpetas nuevas con `--salida` y pásalas al
ejecutable general mediante `--entrada`. El lanzador `experimentos.todas` utiliza
los ocho nombres de la tabla. Las copias incluyen el modelo original sin cambios,
la matriz alineada y un resumen con las huellas de origen y los usuarios excluidos.
Cada usuario conserva su partición y número de fila originales en `usuarios.json`.
