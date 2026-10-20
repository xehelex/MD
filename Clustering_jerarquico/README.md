# Clustering jerárquico aglomerativo

Implementación propia del clustering jerárquico aglomerativo, sin librerías de clustering (ni sklearn ni
`scipy.cluster`). Enlaces **single, complete, average y ward** y distancias **coseno, euclídea, Manhattan,
Chebyshev y Minkowski**.

```
Clustering_jerarquico/
├── __main__.py       # punto de entrada: modos experimentar y modelo
├── algoritmo.py      # distancias, ClusteringJerarquico, ResultadoJerarquico y ModeloEntrenado (el .pkl)
├── metricas.py       # silhouette, cofenética, Davies-Bouldin, Calinski-Harabasz, ARI
├── experimentos.py   # búsqueda del mejor k para cada distancia y enlace, tabla y sugerencia
├── plots.py          # dendrograma, métricas por k y proyección MDS
├── config.json
└── requirements.txt
```

## Requisitos

- Python 3.10 o superior (probado con 3.13).
- `numpy`, `scipy` (solo para matrices dispersas), `matplotlib` (gráficas) y `tqdm` (barra de progreso):
  `pip install -r Clustering_jerarquico/requirements.txt`.
- El paquete `vectorizacion` al lado de este, porque la matriz se carga con
  `vectorizacion.matrices.cargar_matriz`. La entrada es una carpeta con `X.npz` (dispersa, TF-IDF) o
  `X.npy` (densa, SVD).
- Se guarda la matriz de distancias completa (n x n), así que no está pensado para decenas de miles de instancias.

## Uso

Desde la carpeta raíz del proyecto (la que contiene `Clustering_jerarquico/` y `vectorizacion/`):

```bash
python -m Clustering_jerarquico                 # pregunta: 1) experimentar  2) generar modelo.pkl
python -m Clustering_jerarquico experimentar
python -m Clustering_jerarquico modelo
python -m Clustering_jerarquico experimentar --entrada resultados_tfidf_completo_svd95pct --salida resultados_clustering_svd95pct
```

1. **experimentar**: para cada distancia y enlace ajusta el árbol una vez, busca el mejor k con cada criterio, imprime una tabla y sugiere el bloque `mejor_modelo`.
2. Elegimos el modelo y lo escribimos en `mejor_modelo` de `config.json`.
3. **modelo**: entrena esa configuración y guarda `<salida>/modelo.pkl`.

## Flujo de datos

```
X.npz / X.npy ──> cargar_matriz ──> X (n usuarios x d atributos)
                                     │
        experimentar ────────────────┤ para cada distancia x enlace:
                                     │   distancias(X, X) ──> matriz D (n x n)
                                     │   ClusteringJerarquico.fit ──> árbol (n-1 fusiones)
                                     │   buscar_k: etiquetas(k) + evaluar_corte(k) ──> mejor k por criterio
                                     │   ──> CSV, JSON, etiquetas .npy y gráficas
                                     │
        modelo ──────────────────────┘ entrenar_modelo(X, mejor_modelo) ──> modelo.pkl
                                                                           │
        usuarios nuevos ──> mismo vectorizador ──> ModeloEntrenado.predecir ┘ ──> cluster
```

## Configuración (`config.json`)

| Clave | Descripción |
|---|---|
| `entrada`, `salida` | Carpetas relativas a la raíz del proyecto (`--entrada` y `--salida` tienen prioridad). |
| `distancias` | Lista de distancias que se prueban. La sugerencia usa la primera que sea `cosine` o `euclidean`. |
| `enlaces` | Lista con `single`, `complete`, `average`, `ward`. |
| `p` | Exponente de la distancia Minkowski. |
| `k_min`, `k_max` | Rango de k (`k_max` se recorta a n-1). |
| `criterios` | `silhouette`, `davies_bouldin`, `calinski_harabasz`. El primero decide la sugerencia, las etiquetas guardadas y las gráficas. |
| `busqueda` | `completa` (todos los k), `mallado` (malla geométrica + bisección) o `binaria` (bisección pura, solo fiable con un único máximo). |
| `puntos_malla` | Puntos de la malla de `mallado`. |
| `comprobar` | Evalúa también todos los k y dice si la búsqueda encontró el mismo k. |
| `repeticiones_estabilidad` | Submuestreos del 80 % para medir la estabilidad (ARI). 0 = no se calcula. |
| `semilla` | Semilla de los submuestreos. |
| `graficas` | Si es `false` no se generan gráficas. |
| `mejor_modelo` | `enlace`, `distancia` (`cosine` o `euclidean`), `k` y `archivo` del modelo final. |

## Cómo leer la tabla

- **Tamaños**: cluster más pequeño / más grande. `*` = algún cluster con menos del 1 % de los usuarios (casi siempre outliers, no grupos).
- `~` = **corte ambiguo**: las fusiones a ambos lados del corte tienen la misma altura, así que la partición depende del desempate (pasa con `complete` sobre TF-IDF, donde muchas distancias valen exactamente 1).
- La sugerencia ignora las filas con `*` o `~` y solo compara enlaces con la misma distancia. Si el silhouette es menor que 0.25 avisa de que los grupos son poco claros.
- **Evaluac.**: cuántos k se han evaluado de todo el rango (los criterios comparten los k ya calculados).

## Resultados

```
<salida>/
├── evaluaciones_k.csv                      # métricas de cada k evaluado
├── resumen_mejor_k.csv / resultados.json   # mejor k por distancia, enlace y criterio
├── etiquetas_<distancia>_<enlace>_k<k>.npy
├── plots/<distancia>_<enlace>/             # dendrograma, metricas_por_k y mds_2d
└── modelo.pkl                              # solo con el modo "modelo"
```

## El `modelo.pkl`

```python
from Clustering_jerarquico import ModeloEntrenado

modelo = ModeloEntrenado.cargar("resultados_clustering_completo/modelo.pkl")
print(modelo.resumen())
modelo.etiquetas              # cluster de cada usuario de entrenamiento
modelo.predecir(X_nuevo)      # cluster de usuarios nuevos (mismos atributos que el entrenamiento)
```

`predecir` asigna cada fila nueva al cluster al que menos le costaría unirse según el enlace del modelo; el
árbol no se recalcula. Las filas nuevas tienen que venir del mismo vectorizador que el entrenamiento. Para
cargar el `.pkl` el paquete tiene que poder importarse (hay que ejecutar desde la raíz del proyecto) y solo
deben abrirse archivos `.pkl` de confianza.
