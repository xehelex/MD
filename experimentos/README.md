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
python -m experimentos --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo
```

Repite `--entrada` para comparar varias matrices:

```powershell
python -m experimentos --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_completo
```

El tiempo depende del número de usuarios y de la representación. La estabilidad reconstruye árboles adicionales.

## Opciones

| Opción | Por defecto | |
|---|---|---|
| `--entrada` | obligatoria | Carpeta con `X.npz` o `X.npy` y `usuarios.json`. Repetible. |
| `--salida` | `resultados_experimentos` | Dónde dejar la tabla. |
| `--k-min` / `--k-max` | 2 y 20 | Rango de k. |
| `--minimo` | 10 | Tamaño a partir del cual un grupo cuenta. |
| `--estabilidad` | 0 | Submuestreo al 80% sin reemplazo. Multiplica el tiempo por N. |
| `--bootstrap` | 0 | Remuestreo con reemplazo. Mismo coste. |
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
| `bootstrap_ari` | ARI contra remuestreos con reemplazo. |
| `bootstrap_ancho` | Ancho del intervalo 95%. Estrecho = da siempre lo mismo. |
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

## Consultar un texto nuevo

Asigna un texto a un cluster y enseña los usuarios más parecidos dentro de él.

```powershell
python -m experimentos.consultar "I cannot stop gambling" --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo_svd100
```

```
ward / cosine / k = 10
El texto cae en el cluster 1, con 19 usuarios.

usuario           dist  texto
------------------------------------------------------------------------------
subject3358      0.195  26 M unmarried here. I never used to be a gambler be
subject2325      0.197  Let me tell you my story. I am 31 and I started gamb
```

Sin texto lo pide por teclado. Con `--grafico consulta.png` guarda la nube de
usuarios proyectada a 2D, con el texto marcado y sus vecinos resaltados.

| Opción | Por defecto | |
|---|---|---|
| `--entrada` | obligatoria | La representación con la que se entrenó el modelo. |
| `--modelo` | `resultados_clustering_completo/modelo.pkl` | |
| `--textos` | `output/data_preprocesado.csv` | Para enseñar el texto de los vecinos. |
| `--vecinos` | 5 | |
| `--grafico` | — | Ruta del PNG. |

El vectorizador y el SVD se resuelven solos leyendo el `resumen.json` de la
carpeta, que guarda de dónde viene cada representación. Siempre con `transform`:
reajustar el vocabulario con el texto nuevo invalidaría la comparación.

**La distancia dice cuánto fiarse.** Un vecino a 0,19 significa que el texto ha
encontrado a su gente; a 0,75, que ha caído en el grupo grande por descarte.

El texto necesita al menos 3 palabras con contenido: las vacías (`I`, `can`,
`the`) se descartan en el preproceso, igual que se hizo al entrenar.

El gráfico usa PCA y no t-SNE porque PCA sabe proyectar puntos nuevos sobre un
plano ya calculado; con t-SNE habría que rehacer la nube en cada consulta.

## Bootstrap

Responde a "¿y si me hubieran tocado otros 598 usuarios?". Saca muestras del
propio conjunto **con reemplazo**, rehace el árbol y compara con la partición
original por ARI. Al coger n con reemplazo se queda fuera un 37% por azar, así
que cada vuelta es un conjunto del mismo tamaño con distinta composición.

```powershell
python -m experimentos.bootstrap --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo_svd100 --k 7
```

```
enlace       ARI     intervalo 95%   ancho  tamanos
--------------------------------------------------------------------------
single     0.441   [0.177, 0.796]   0.619  [592, 1, 1, 1, 1, 1, 1]
average    0.530   [0.297, 0.801]   0.504  [577, 7, 5, 4, 2, 2, 1]
complete   0.496   [0.303, 0.634]   0.332  [447, 71, 28, 25, 11, 9, 7]
ward       0.432   [0.291, 0.559]   0.268  [231, 123, 92, 85, 30, 20, 17]
```

| Opción | Por defecto | |
|---|---|---|
| `--entrada` | obligatoria | Carpeta con `X.npz` o `X.npy`. |
| `--k` | obligatoria | |
| `--repeticiones` | 50 | 30 para mirar, 100 para la memoria. |
| `--enlace` | los cuatro | Repetible. |
| `--distancia` | `cosine` | |

**La media engaña igual que el silhouette: `average` gana con 577/7/5/4/2/2/1.**
Una partición degenerada saca ARI alto porque es trivial de repetir, no porque
sea buena. Por eso está la columna de tamaños al lado.

Lo que informa es el **ancho**. Ward saca la nota más baja y la saca siempre
(0,268); single va de 0,177 a 0,796, o sea que depende por completo de qué
usuarios le toquen. Para la memoria, "ward, ARI 0,43 ± 0,13" es defendible;
"average, 0,53" sin el intervalo es hacer trampa.

Con `single` el número sale inflado aparte de eso: los usuarios repetidos quedan
a distancia 0 entre sí y le regalan puentes para encadenar.

### En la rejilla

Para comparar los cuatro enlaces por todos los k de una pasada, el banco lo trae
como dos columnas más:

```powershell
python -m experimentos --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo_svd100 --bootstrap 50
```

El árbol de cada remuestreo se construye una vez y se corta en todos los k, igual
que hace `--estabilidad`. Si no, no entra: con 50 remuestreos × 19 valores de k ×
5 combinaciones serían casi 5.000 árboles en vez de 250.

Da exactamente lo mismo que el comando suelto, y hay un test que lo comprueba
([test_bootstrap.py](test_bootstrap.py)) porque es justo donde una optimización así
se rompe sin avisar.

### Desde Python, si hace falta el histograma

```python
from experimentos.bootstrap import bootstrap

r = bootstrap(X, "ward", "cosine", k=7, repeticiones=50)
print(r.resumen(), r.ancho())
print(r.aris)           # un valor por repetición
```

## Pruebas

```powershell
python -m unittest discover -s experimentos -t .
```

## Matrices completas

Cada entrada contiene todos los usuarios disponibles tras el preprocesamiento. Las carpetas antiguas con train/test se conservan como resultados históricos; no se cargan automáticamente. La estabilidad usa submuestreo sin reemplazo sobre vectores fijos, sin reajustar TF-IDF ni SVD. Al comparar se comprueba que las representaciones tengan los mismos usuarios y orden de filas.

## Comparar TF-IDF con SVD

Primero genera SVD siguiendo el README de `vectorizacion`. Luego:

```powershell
python -m experimentos --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo_svd100 --salida resultados_experimentos_svd_palabras
python -m experimentos --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_completo --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_completo_svd100 --salida resultados_experimentos_svd_caracteres
```

Las matrices SVD se cargan como arrays densos, sin convertirlas a dispersas.
Las métricas y el remuestreo son los mismos que para TF-IDF.
