# Banco de pruebas

Para comparar las **ocho representaciones preparadas**, sigue
[GUIA_TODAS.md](GUIA_TODAS.md). Comprobación rápida desde la raíz de `MD`:

```powershell
python -m experimentos.todas --comprobar-entradas
python -m experimentos.todas --estabilidad 5 --salida resultados_experimentos_todas_estabilidad
```

Compara configuraciones de clustering jerárquico sobre una o varias matrices ya
vectorizadas y saca una tabla con la calidad de cada partición. Solo lee los
`X.npz` (dispersa) o `X.npy` (densa) que le indiques; no reajusta la vectorización.
Cada carpeta debe contener exactamente una matriz y su `usuarios.json`.

## Ejecutar

Desde la carpeta `MD`:

```powershell
python -m pip install -r experimentos/requirements.txt
python -m experimentos --entrada resultados_tfidf_completo
```

Repite `--entrada` para comparar varias matrices:

```powershell
python -m experimentos --entrada resultados_tfidf_completo --entrada resultados_tfidf_caracteres4_completo
```

El tiempo depende del número de usuarios y de la representación. La estabilidad reconstruye árboles adicionales.

## Opciones

| Opción | Por defecto | |
|---|---|---|
| `--entrada` | obligatoria | Carpeta con `X.npz` o `X.npy` y `usuarios.json`. Repetible. |
| `--salida` | `resultados_experimentos` | Dónde dejar la tabla. |
| `--k-min` / `--k-max` | 2 y 20 | Rango de k. |
| `--minimo` | 10 | Tamaño a partir del cual un grupo cuenta. |
| `--estabilidad` | 0 | Repeticiones de remuestreo. Multiplica el tiempo por N. |
| `--comprobar-entradas` | desactivada | Valida matrices e IDs sin ejecutar clustering ni guardar resultados. |

## Salida

`rejilla.csv`, una fila por matriz × enlace × k:

| Columna | |
|---|---|
| `silhouette` | Métrica interna clásica. |
| `equilibrio` | Entropía de los tamaños, 0 a 1. 1 = grupos iguales. |
| `mayor_pct` | % de instancias en el grupo mayor. |
| `grupos_utiles` | Grupos por encima de `--minimo`. |
| `instancias_sueltas` | Las que caen por debajo. |
| `tamanos` | Los seis grupos mayores. |
| `estabilidad` | ARI contra un reajuste al 80%. |
| `inconsistencia` | Cuánto destaca esa fusión frente a su vecindario. |
| `cofenetica` | Distancias originales contra alturas del árbol. |
| `segundos_arbol` | Coste de construir el árbol. |

Y `resumen.json` con los parámetros y la correlación silhouette-equilibrio.

**El `equilibrio` va al lado del `silhouette` porque el silhouette solo no
distingue una partición repartida de un 477/1/1/1: de hecho puntúa mejor la
segunda.** Por el mismo motivo el resumen por pantalla enseña el k que elige el
silhouette y el que propone el árbol, uno al lado del otro.

## Los otros dos módulos

```python
from experimentos.corte import cortar_por_tamano
from experimentos.eleccion_k import sugerir_k

sugerir_k(resultado)                                  # [(10, 3.13), (9, 2.4), ...]
cortar_por_tamano(resultado, minimo=8, max_grupos=10) # (etiquetas, informe)
```

`sugerir_k` lee la forma del árbol en vez de barrer k. Da candidatos, no una
respuesta: con datos sintéticos de k conocido acierta en el top-2, no siempre
en el primero.

`cortar_por_tamano` pide grupos de al menos `minimo` miembros y aparta el resto
con etiqueta `-1`. Ojo con el umbral: subirlo borra los grupos pequeños, y aquí
los pequeños suelen ser los interesantes.

## Pruebas

```powershell
python -m unittest discover -s experimentos -t .
```

## Matrices completas

Cada entrada contiene todos los usuarios disponibles tras el preprocesamiento. Las carpetas antiguas con train/test se conservan como resultados históricos; no se cargan automáticamente. La estabilidad usa submuestreo sin reemplazo sobre vectores fijos, sin reajustar TF-IDF ni SVD. Al comparar se comprueba que las representaciones tengan los mismos usuarios y orden de filas.

## Comparar TF-IDF con SVD

Primero genera SVD siguiendo el README de `vectorizacion`. Luego:

```powershell
python -m experimentos --entrada resultados_tfidf_completo --entrada resultados_tfidf_completo_svd100 --salida resultados_experimentos_svd_palabras
python -m experimentos --entrada resultados_tfidf_caracteres4_completo --entrada resultados_tfidf_caracteres4_completo_svd100 --salida resultados_experimentos_svd_caracteres
```

Las matrices SVD se cargan como arrays densos, sin convertirlas a dispersas.
Las métricas y el remuestreo son los mismos que para TF-IDF.
