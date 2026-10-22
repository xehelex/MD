# Vectorización densa con transformers

Convierte las publicaciones de cada usuario en un vector denso. A diferencia de TF-IDF, el vector de cada
palabra depende de su contexto, así que dos usuarios que hablan de lo mismo con palabras distintas pueden
quedar cerca.

Son dos pasos:

1. **Embeddings**: un vector por usuario, la media de los vectores de sus posts.
2. **Reducción**: PCA + UMAP con los parámetros de `config.json`, para dejarlo listo para el clustering.

```
vectorizacion_densa/
├── __main__.py        # embeddings (si no existen) + reducción
├── roberta.py         # RobertaPorUsuario: limpieza, vector por post y media por usuario
├── reduccionDim.py    # solo la reducción PCA + UMAP
├── config.json        # conjunto, modelo y parámetros de PCA y UMAP
├── requirements.txt
└── test_roberta.py    # vacío por ahora
```

## Modelos

| `--modelo` | Modelo de Hugging Face | Dimensiones | Tokens por post |
|---|---|---:|---:|
| `roberta` | `FacebookAI/roberta-large` | 1024 | 512 |
| `all-roberta` | `sentence-transformers/all-roberta-large-v1` | 1024 | 256 |
| `multilingue` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 128 |

## Requisitos

- Python 3.10 o superior y `pip install -r vectorizacion_densa/requirements.txt`.
- Una GPU NVIDIA es muy recomendable: en CPU puede tardar horas. En Windows, `pip install torch` instala la
  versión sin GPU; para usar CUDA hay que instalar torch con el comando que da [pytorch.org](https://pytorch.org).
- La primera vez se descarga el modelo de Hugging Face (`roberta-large` ocupa unos 1,4 GB).

## Configuración (`config.json`)

| Clave | Descripción |
|---|---|
| `conjunto` | `datos_1_a_600` o `datos_completos`, como en `resultados_tfidf/`. |
| `modelo` | `roberta`, `all-roberta` o `multilingue`. |
| `reduccion.varianza` | Varianza que conserva PCA, entre 0 y 1 (0.9 = 90 %). |
| `reduccion.min_dist` | `min_dist` de UMAP, entre 0 y 1. Con 0 los grupos quedan más apretados. |
| `reduccion.componentes` | Dimensiones finales de UMAP. |
| `reduccion.vecinos` | `n_neighbors` de UMAP. |
| `reduccion.semilla` | Semilla de UMAP. |
| `conjuntos` | Para cada conjunto, el CSV de entrada (`datos`) y el `usuarios.json` de TF-IDF con el que se alinean los usuarios. |

`--conjunto` y `--modelo` en la línea de comandos tienen prioridad sobre el JSON.

## Uso

Desde la raíz de `MD`, con el menú (`python main.py` → Vectorizar → 4 o 5) o directamente:

```powershell
.\.venv\Scripts\python.exe -m vectorizacion_densa --conjunto datos_1_a_600 --modelo roberta
```

Calcula los embeddings si no existen y después aplica PCA + UMAP. Otras opciones:

- `--sin-reduccion`: solo calcula los embeddings.
- `python -m vectorizacion_densa.reduccionDim --conjunto ... --modelo ...`: solo repite la reducción, por
  ejemplo después de cambiar `min_dist` en el JSON. No carga el modelo ni necesita GPU.

Los embeddings se reutilizan si ya existen. Para recalcularlos, borra su carpeta `sin_reduccion`.

## Resultados

```
resultados_roberta/
├── datos_1_a_600/            598 usuarios (los mismos que en resultados_tfidf/datos_1_a_600)
│   └── <modelo>/
│       ├── sin_reduccion/                    embeddings tal cual
│       └── PCA_umap_<varianza>_<min_dist>/   p. ej. PCA_umap_0.9_0.0
└── datos_completos/          2.344 usuarios (los de resultados_tfidf/datos_completos)
```

El nombre de la carpeta reducida sale de `varianza` y `min_dist` del JSON. Si se cambian otros parámetros y no
esos dos, se sobrescribe la misma carpeta; los parámetros usados quedan siempre en su `config.json`.

| Archivo | Contenido |
|---|---|
| `X.npy` | Matriz usuarios × dimensiones (float32), la que lee el clustering. |
| `usuarios.json` | `fila`, `usuario` y `n_posts` de cada fila, con el mismo formato que TF-IDF. |
| `user_embeddings.npz` | La misma matriz junto con `subject_id` y `n_posts`. |
| `config.json` | Conjunto, modelo, CSV de entrada y parámetros de la reducción. |
| `reductor.joblib` | Solo en las reducidas: escalado + PCA + UMAP ajustados, para transformar usuarios nuevos. |

## Cómo funciona

1. **Limpieza ligera**: junta título y texto, deshace las entidades HTML (`&amp;`) y quita URLs, menciones
   (`u/usuario`, `@usuario`) y espacios repetidos. No quita stopwords ni lematiza: el modelo necesita el
   texto natural para entender el contexto. Por eso la entrada es `data.csv` y no el preprocesado.
2. **Un vector por post**: media de los vectores de sus tokens en la última capa, sin contar el relleno.
   Con `all-roberta` se normaliza además a norma 1.
3. **Un vector por usuario**: la media de los vectores de sus posts.
4. **Alineación**: se quedan los usuarios del `usuarios.json` de TF-IDF del mismo conjunto y en su orden.
   Se descartan los que el preprocesado dejó sin texto útil (2 en `datos_1_a_600`).
5. **Reducción**: estandariza, PCA hasta la varianza del JSON y UMAP con distancia coseno.

Para que vaya más rápido, los textos repetidos se calculan una sola vez, los posts se ordenan por longitud
(así cada lote lleva poco relleno) y en GPU se usa media precisión (fp16).

## Usarlo en el clustering y en el banco de pruebas

```powershell
.\.venv\Scripts\python.exe -m Clustering_jerarquico experimentar --entrada resultados_roberta/datos_1_a_600/roberta/PCA_umap_0.9_0.0 --salida resultados_clustering_roberta
.\.venv\Scripts\python.exe -m experimentos --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_completo_svd100 --entrada resultados_roberta/datos_1_a_600/roberta/PCA_umap_0.9_0.0
```

Como los usuarios están alineados con TF-IDF, el banco puede comparar las dos representaciones.

**Ojo con UMAP:** con `min_dist` = 0 apelotona los puntos, y el silhouette medido en ese espacio sale más
alto que en el espacio original. Para comparar con TF-IDF, mide también la partición sobre `sin_reduccion`.
