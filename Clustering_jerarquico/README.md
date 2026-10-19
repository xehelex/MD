# Clustering jerárquico aglomerativo

Implementación propia (NumPy + SciPy, sin librerías de clustering) de clustering jerárquico aglomerativo con enlaces **single, complete, average y ward**, distancias **coseno, euclídea, Manhattan, Chebyshev y Minkowski**, un banco de métricas para elegir el número de clusters y la generación de un **`modelo.pkl`** con el mejor modelo.

## Estructura

```
Clustering_jerarquico/
├── __init__.py       # convierte la carpeta en paquete y exporta la API
├── __main__.py       # punto de entrada: configuración, menú y modos experimentar / modelo
├── algoritmo.py      # ClusteringJerarquico, ResultadoJerarquico y ModeloEntrenado (el .pkl)
├── metricas.py       # silhouette, cofenética, Davies-Bouldin, Calinski-Harabasz, ARI
├── plots.py          # dendrograma, curva de silhouette, MDS 2D, búsqueda de k
├── experimentos.py   # clustering por enlace, búsqueda del mejor k y tablas comparativas
├── config.json       # configuración
├── requirements.txt
└── README.md
```

## Instalación

```bash
pip install -r Clustering_jerarquico/requirements.txt
```

Requiere Python 3.10 o superior (se usan anotaciones como `X | None`).

## Datos de entrada

Una matriz completa dispersa TF-IDF (`X.npz`) o densa SVD (`X.npy`) en:

```
<carpeta_padre_del_paquete>/<entrada>/X.npz
# o bien X.npy; solo una de las dos en cada carpeta
```

Se carga con `vectorizacion.matrices.cargar_matriz`, así que hay que ejecutar desde la carpeta raíz del proyecto (la que contiene `Clustering_jerarquico/` y `vectorizacion/`).

Las rutas `entrada` y `salida` de `config.json` son relativas a la carpeta que **contiene** el paquete.
El valor predeterminado es `resultados_tfidf_completo`, con 598 usuarios.
Las carpetas antiguas train/test se conservan como históricos; para usarlas en
clustering se emplean sus copias `*_historico_alineado`.

## Flujo de trabajo

1. **Experimentar**: `python -m Clustering_jerarquico experimentar`. Ajusta cada enlace, busca el mejor k con tres criterios e imprime una tabla comparativa con una sugerencia.
2. **Elegir** el mejor modelo (tú decides) y escribirlo a mano en el bloque `mejor_modelo` de `config.json`.
3. **Generar el modelo**: `python -m Clustering_jerarquico modelo`. Entrena esa configuración y guarda `modelo.pkl`.

## Uso

Desde la carpeta que contiene el paquete (la que está un nivel por encima de `Clustering_jerarquico/`):

```bash
python -m Clustering_jerarquico                 # pregunta: 1) experimentar  2) generar modelo.pkl
python -m Clustering_jerarquico experimentar    # clustering por enlace + búsqueda del mejor k
python -m Clustering_jerarquico modelo          # entrena el "mejor_modelo" de config.json y guarda modelo.pkl
python -m Clustering_jerarquico experimentar --entrada resultados_tfidf_completo_svd95pct --salida resultados_clustering_svd95pct
python -m Clustering_jerarquico experimentar --distancias cosine euclidean --enlaces average ward --comprobar
```

Sin modo y sin más opciones muestra el menú. Si se pasan opciones sin modo (por ejemplo solo `--entrada` y `--salida`, como antes) se asume `experimentar`. También se puede lanzar con el botón Run de PyCharm sobre `__main__.py`.

Para recorrer las ocho representaciones con el banco de pruebas y estabilidad,
consulta [la guía conjunta](../experimentos/GUIA_TODAS.md).

### Qué hace `experimentar`

1. **Clustering por enlace**: ajusta un modelo por cada enlace de `config.json` (incluido `ward`), calcula el silhouette para cada k entre `k_min` y `k_max` y se queda con el mejor. Guarda etiquetas, gráficas y `resumen_metricas.json`.
2. **Búsqueda del mejor k**: para cada distancia y enlace busca el mejor k con silhouette, Davies-Bouldin y Calinski-Harabasz sin recorrer todo el rango.
3. **Resultados por pantalla**: dos tablas (una por fase) y la sugerencia de `mejor_modelo` lista para pegar en `config.json`.

Cómo leer las tablas y la sugerencia:

- **Tamaños** es el cluster más pequeño y el más grande. Un `*` marca particiones con algún cluster de menos del 1 % de los usuarios (mínimo 2): casi siempre es un usuario aislado (outlier), no un grupo.
- Un `~` marca un **corte ambiguo**: las fusiones a ambos lados del corte tienen exactamente la misma altura (en `complete` con coseno pasa a menudo, porque muchas distancias valen 1.0), así que la partición depende del orden de desempate.
- La **sugerencia ignora** las filas con `*` o `~` y solo compara enlaces dentro de una misma distancia. Es un punto de partida: la decisión es tuya.
- Si el mejor silhouette es menor que 0.25 se imprime un aviso: con esa representación los datos casi no forman grupos y conviene comparar con otras representaciones (por ejemplo `--entrada resultados_tfidf_caracteres4_completo_svd100`).

### Qué hace `modelo`

Lee `mejor_modelo` de `config.json`, ajusta el clustering con esa configuración, corta el árbol en `k` clusters y guarda un `ModeloEntrenado` en `<salida>/<archivo>`. Imprime sus métricas y cómo cargarlo.

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
| `experimentos.busqueda` | `mallado` (por defecto) o `binaria`. |
| `experimentos.puntos_malla` | Puntos de la malla geométrica gruesa. |
| `experimentos.comprobar` | Evalúa todos los k y compara con la búsqueda (más lento). |
| `experimentos.criterios` | Lista con `silhouette`, `davies_bouldin`, `calinski_harabasz`. El primero decide la sugerencia. |
| `mejor_modelo.enlace` | Enlace del modelo final. |
| `mejor_modelo.distancia` | `cosine` o `euclidean`. |
| `mejor_modelo.k` | Número de clusters, entero con `2 <= k < n`. |
| `mejor_modelo.archivo` | Nombre del `.pkl`, dentro de la carpeta `salida` (opcional, por defecto `modelo.pkl`). |

La clave antigua `ejecutar_experimentos` ya no hace falta: `experimentar` siempre ejecuta las dos fases.

Las opciones de línea de comandos tienen prioridad sobre el JSON. `--distancias` solo afecta a la búsqueda de k; el clustering por enlace usa la `distancia` de `config.json`.

## El `modelo.pkl`

Contiene el árbol de fusiones, la matriz de entrenamiento, las etiquetas, los centroides, las métricas y la fecha de creación.

```python
from Clustering_jerarquico import ModeloEntrenado

modelo = ModeloEntrenado.cargar("resultados_clustering_completo/modelo.pkl")
print(modelo.resumen())          # configuración y métricas
modelo.etiquetas                 # cluster de cada usuario de entrenamiento
modelo.predecir(X_nuevo)         # cluster de usuarios nuevos (mismos atributos que el entrenamiento)
```

El clustering jerárquico no predice de forma nativa. `predecir` asigna cada fila nueva al cluster al que menos le costaría unirse, con la regla del enlace del modelo:

| Enlace | Regla |
|---|---|
| single | distancia mínima a los miembros del cluster |
| complete | distancia máxima |
| average | distancia media |
| ward | aumento de la suma de cuadrados: `n_c/(n_c+1) · ‖x − centroide_c‖²` |

Es una aproximación: el árbol no cambia al añadir filas. Además, si se vuelven a predecir las propias filas de entrenamiento, la asignación coincide con las etiquetas del corte del árbol en torno al 90-95 % de los casos, no en el 100 %: el corte de un dendrograma no es una partición de Voronoi y algunas filas quedan más cerca del centro del otro cluster. Las filas nuevas tienen que estar en el mismo espacio de atributos (mismo vectorizador). Para cargar el `.pkl` el paquete `Clustering_jerarquico` tiene que ser importable (ejecuta desde la carpeta raíz del proyecto), y solo hay que cargar archivos `.pkl` de confianza.

## Ward y distancia coseno

Ward solo tiene sentido con distancia euclídea. Cuando la configuración pide `cosine` con `ward`, el programa normaliza las filas a norma 1 y aplica Ward con distancia euclídea. Sobre vectores unitarios `d_euclídea² = 2 · d_coseno`, de modo que el resultado respeta la geometría coseno.

En los resultados, `distancia` indica lo que se pidió y `distancia_efectiva` lo que se calculó realmente (`euclidean` para Ward). **El silhouette de Ward se evalúa con la distancia pedida (coseno)**, no con la euclídea interna, para que sea comparable con el de single, complete y average. Davies-Bouldin y Calinski-Harabasz se calculan para todos sobre filas normalizadas.

## Métricas

| Métrica | Mejor | Notas |
|---|---|---|
| Silhouette medio | mayor | Usa la matriz de distancias de la distancia pedida. |
| Davies-Bouldin | menor | Euclídea sobre centroides. Densifica X, así que conviene en datasets pequeños o medianos. |
| Calinski-Harabasz | mayor | Dispersión entre clusters / dispersión interna. No densifica X (solo materializa los k centroides). |
| Correlación cofenética | mayor | Correlación de Pearson entre las distancias originales y las alturas del árbol. |
| ARI | cerca de 1 | Para medir estabilidad entre submuestreos. |

Con distancia coseno, Davies-Bouldin y Calinski-Harabasz se calculan sobre las filas normalizadas. Los valores de métricas solo se pueden comparar entre enlaces que usan la misma distancia.

## Resultados

```
<salida>/
├── modelo.pkl                       # solo con el modo "modelo"
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
- El `.pkl` guarda la matriz de entrenamiento y la de distancias (n²): con 598 usuarios son unos pocos MB.
- `asignar_nuevo` y `predecir` son aproximaciones, no una predicción nativa del árbol.
