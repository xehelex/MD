# Clustering jerárquico aglomerativo

Implementación propia (NumPy + SciPy, sin librerías de clustering) de clustering jerárquico aglomerativo con enlaces **single, complete, average y ward**, distancias **coseno, euclídea, Manhattan, Chebyshev y Minkowski**, y un banco de métricas para elegir el número de clusters.

## Estructura

```
Clustering_jerarquico/
├── __init__.py          # API pública del paquete
├── __main__.py          # python -m Clustering_jerarquico
├── algoritmo.py         # ClusteringJerarquico y ResultadoJerarquico
├── metricas.py          # silhouette, cofenética, Davies-Bouldin, Calinski-Harabasz, ARI
├── plots.py             # dendrograma, curva de silhouette, MDS 2D, búsqueda de k
├── cli.py               # ejecución completa a partir de config.json
├── experimentos.py      # búsqueda del mejor k (se lanza también desde el CLI)
├── validacion.py        # contraste contra scipy
├── test_algoritmo.py    # pruebas unitarias
├── config.json          # configuración
└── requirements.txt
```

## Instalación

```bash
pip install -r requirements.txt
```

Requiere Python 3.10 o superior (se usan anotaciones como `X | None`).

## Datos de entrada

Una matriz dispersa (por ejemplo TF-IDF) guardada con `scipy.sparse.save_npz` en:

```
<carpeta_padre_del_paquete>/<entrada>/X_train.npz
```

Las rutas `entrada` y `salida` de `config.json` son relativas a la carpeta que **contiene** el paquete.

## Uso

Desde la carpeta que contiene el paquete (la que está un nivel por encima de `Clustering_jerarquico/`):

```bash
python -m Clustering_jerarquico                # clustering + experimentos (si está activado)
python -m Clustering_jerarquico.experimentos   # solo experimentos, con argumentos propios
python -m unittest Clustering_jerarquico.test_algoritmo
```

### Qué hace `python -m Clustering_jerarquico`

1. Ajusta un modelo por cada enlace de `config.json` (incluido `ward`).
2. Calcula el silhouette para cada k entre `k_min` y `k_max` y se queda con el mejor.
3. Guarda etiquetas, gráficas y `resumen_metricas.json` en la carpeta de salida.
4. Si `"ejecutar_experimentos": true`, ejecuta a continuación `experimentos.py` con los valores del bloque `experimentos` de la configuración.

## Configuración (`config.json`)

| Clave | Descripción |
|---|---|
| `entrada`, `salida` | Carpetas de entrada y salida (relativas a la carpeta que contiene el paquete). |
| `distancia` | `cosine` o `euclidean`. |
| `enlaces` | Lista con `single`, `complete`, `average`, `ward`. |
| `k_min`, `k_max` | Rango de k. Debe cumplirse `2 <= k_min <= k_max < n`. |
| `calcular_estabilidad` | Si es `true`, calcula la estabilidad (ARI) con submuestreos del 80 %. |
| `repeticiones_estabilidad` | Número de submuestreos (solo si `calcular_estabilidad` es `true`). |
| `semilla` | Semilla del generador aleatorio de la estabilidad. |
| `generar_graficas` | Si es `false`, no se generan dendrograma, silhouette ni MDS. |
| `incluir_db` | Calcula también Davies-Bouldin para el k final. |
| `ejecutar_experimentos` | Lanza `experimentos.py` al terminar el CLI. |
| `experimentos.busqueda` | `mallado` (por defecto) o `binaria`. |
| `experimentos.puntos_malla` | Puntos de la malla geométrica gruesa. |
| `experimentos.comprobar` | Evalúa todos los k y compara con la búsqueda (más lento). |
| `experimentos.criterios` | Lista con `silhouette`, `davies_bouldin`, `calinski_harabasz`. |

Los argumentos de línea de comandos de `experimentos.py` tienen prioridad sobre el bloque `experimentos`. Ejemplo:

```bash
python -m Clustering_jerarquico.experimentos --distancias cosine euclidean --enlaces average ward --comprobar
```

## Ward y distancia coseno

Ward solo tiene sentido con distancia euclídea. Cuando la configuración pide `cosine` con `ward`, el programa normaliza las filas a norma 1 y aplica Ward con distancia euclídea. Sobre vectores unitarios `d_euclídea² = 2 · d_coseno`, de modo que el resultado respeta la geometría coseno.

Consecuencia: en los resultados, `distancia` indica lo que se pidió y `distancia_efectiva` lo que se calculó realmente (`euclidean` para Ward). Los silhouette de Ward se calculan con esa distancia euclídea, por lo que **no son directamente comparables** con los de los otros enlaces, que usan coseno.

## Métricas

| Métrica | Mejor | Notas |
|---|---|---|
| Silhouette medio | mayor | Usa la matriz de distancias del modelo. |
| Davies-Bouldin | menor | Euclídea sobre centroides. Densifica X, así que conviene en datasets pequeños o medianos. |
| Calinski-Harabasz | mayor | Dispersión entre clusters / dispersión interna. No densifica X (solo materializa los k centroides). |
| Correlación cofenética | mayor | Correlación de Pearson entre las distancias originales y las alturas del árbol. |
| ARI | cerca de 1 | Para medir estabilidad entre submuestreos. |

Con distancia coseno, Davies-Bouldin y Calinski-Harabasz se calculan sobre las filas normalizadas.

## Resultados

```
<salida>/
├── etiquetas_<enlace>_k<k>.npy      # etiquetas con el mejor k por silhouette
├── resumen_metricas.json            # métricas por enlace
├── plots/<enlace>/                  # dendrograma, silhouette_por_k, MDS 2D
└── experimentos/<fecha>_<busqueda>/
    ├── evaluaciones_k.csv           # métricas de cada k evaluado
    ├── resumen_mejor_k.csv          # mejor k por distancia, enlace y criterio
    ├── resultados.json
    └── plots/busqueda_k_<distancia>_<enlace>.png
```

## Límites conocidos

- Se construye la matriz de distancias completa: memoria O(n²). Con decenas de miles de instancias no es viable.
- `fit` guarda las distancias en un diccionario de n(n-1)/2 entradas, bastante más pesado que una matriz NumPy; el ajuste está pensado para cientos o pocos miles de instancias.
- Manhattan, Chebyshev y Minkowski con p ≠ 2 densifican X y cuestan O(n²·d).
- `asignar_nuevo` es una aproximación (cluster con menor distancia media), no una predicción nativa del árbol.
