# Vectorización completa: un vector por usuario

Los resultados están en `resultados_tfidf/`, en dos conjuntos: `datos_1_a_600`
(lo que describe este README) y `datos_completos` (las cinco carpetas de CLEF,
2.344 usuarios). Los comandos y cifras del conjunto completo están en
[resultados_tfidf/README.md](../resultados_tfidf/README.md).

Para ejecutar experimentos con las **ocho representaciones**, incluidas las dos
históricas alineadas, consulta [la guía para los compañeros](../experimentos/GUIA_TODAS.md).

Ejecuta los comandos desde la raíz del proyecto `MD`. Se lee el CSV del
preprocesador y se genera una matriz con todos sus usuarios útiles. No hay
separación train/test: el vocabulario y los pesos IDF se aprenden con todos ellos.

## 1. Preparar Python

Necesitas Python 3.10 o superior. La primera vez:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r vectorizacion/requirements.txt
```

Si ya tienes un entorno con las dependencias, puedes usar su Python.

Los modelos SVD `svd.joblib` se guardan en Git LFS. Si vas a cargarlos con
`joblib.load`, instala Git LFS y ejecuta `git lfs install` y `git lfs pull`
desde la raíz de `MD`. Las matrices de experimentos se descargan con Git normal.

## 2. Tener los datos preprocesados

La entrada predeterminada es `output/data_preprocesado.csv`, con las columnas
`subject_id` y `text_Clustering`. Esta última debe contener texto preprocesado
con las palabras separadas por espacios, no listas de Python escritas como texto.

Si todavía no existe el CSV, ejecuta el preprocesador de los compañeros:

```powershell
.\.venv\Scripts\python.exe -m pip install -r preprocesador/requirements.txt
.\.venv\Scripts\python.exe -m preprocesador --datos "C:\ruta\a\data_1_to_600" --guardar_csv
```

Cambia la ruta por la carpeta de tus XML. Los XML se conservan y no se suben a
GitHub. `output/data.csv` conserva los registros extraídos; el segundo CSV añade
el texto procesado y filtra publicaciones según el preprocesador. `output/`
está excluida de Git. La vectorización no vuelve a lematizar ni limpiar el CSV.

## 3. Generar las dos representaciones

**TF-IDF de palabras:**

```powershell
.\.venv\Scripts\python.exe -m vectorizacion --representacion palabras --salida resultados_tfidf/datos_1_a_600/resultados_tfidf_completo
```

**TF-IDF de fragmentos de cuatro caracteres:**

```powershell
.\.venv\Scripts\python.exe -m vectorizacion --representacion caracteres4 --salida resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_completo
```

Para usar otro CSV, añade `--datos "C:\ruta\al\data_preprocesado.csv"`.
Para repetir la ejecución, usa una carpeta nueva, por ejemplo
`--salida resultados_tfidf/datos_1_a_600/resultados_tfidf_completo_v2`. No se sobrescriben carpetas con contenido.

Se mantienen `min_df=3` usuarios, TF lineal, IDF suavizado y normalización L2.
Los resultados se guardan como matrices dispersas float32. Los fragmentos no
cruzan entre publicaciones. Las opciones `--test-size` y `--seed` se han retirado.

## 4. Qué entregar a los compañeros

| Archivo | Contenido |
|---|---|
| `X.npz` | Matriz completa: una fila por usuario. |
| `usuarios.json` | ID y origen de cada fila, en orden lexicográfico de ID. |
| `vectorizador.joblib` | Vocabulario y pesos para representar usuarios nuevos. |
| `resumen.json` | Configuración, huella del CSV, dimensiones, descartes y vectores cero. |

Entrega estas carpetas de resultados y el paquete `vectorizacion`.
Con el mismo CSV, palabras y caracteres conservan los mismos usuarios y orden.
No se garantiza que sean 600: depende de los usuarios que conserve el preprocesador.
El resumen identifica descartes dentro del CSV; no puede recuperar usuarios que
el preprocesador ya haya eliminado antes de exportarlo. Revisar también el aviso
de sujetos sin texto útil que imprime el preprocesador.

Las carpetas antiguas `resultados_tfidf/datos_1_a_600/resultados_tfidf` y `resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4` conservan
los resultados anteriores con train/test. No se han convertido ni sobrescrito.
Sus copias `resultados_tfidf/datos_1_a_600/resultados_tfidf_historico_alineado` y
`resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_historico_alineado` reúnen train y test, conservan
los valores originales y alinean las filas con los 598 usuarios actuales.
El vocabulario y los IDF históricos siguen siendo los ajustados con 480 usuarios.
Los experimentos nuevos leen `X.npz` (dispersa) o `X.npy` (densa); no hay que
renombrar las matrices antiguas. Cada representación debe ir en su propia carpeta.

## 5. Cargar la matriz o transformar usuarios nuevos

```python
import joblib
from scipy.sparse import load_npz
from vectorizacion.tfidf import cargar_historiales_csv

X = load_npz("resultados_tfidf_completo/X.npz")
modelo = joblib.load("resultados_tfidf_completo/vectorizador.joblib")
atributos = modelo.get_feature_names_out()
nuevos = cargar_historiales_csv("otros_usuarios_preprocesados.csv")
X_nuevos = modelo.transform([u.publicaciones for u in nuevos])
```

Los usuarios nuevos deben pasar por el mismo preprocesamiento. No volver a usar
`fit` si se quiere conservar el vocabulario del modelo guardado.

## 6. Ejecutar clustering y comprobaciones

```powershell
.\.venv\Scripts\python.exe -m experimentos --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_completo
.\.venv\Scripts\python.exe -m experimentos --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo --estabilidad 5
.\.venv\Scripts\python.exe -m Clustering_jerarquico --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo
.\.venv\Scripts\python.exe -m unittest vectorizacion.test_tfidf vectorizacion.test_completo
.\.venv\Scripts\python.exe -m vectorizacion --help
```

Los ejecutables de clustering tienen sus dependencias en sus propios
`requirements.txt`. El remuestreo lo realiza el módulo de experimentos. Su
estabilidad actual usa submuestreo del 80% sin reemplazo, no bootstrap clásico,
y evalúa el clustering con la representación ya fijada. Para evaluar todo el
proceso habría que reajustar la vectorización dentro de cada remuestreo.
Para probar usuarios completamente nuevos, deben quedar fuera del ajuste inicial.

Si aparece `No module named vectorizacion`, ejecuta desde la raíz de `MD`.
Si falta el CSV, completa el paso 2 o indica su ruta con `--datos`.

## 7. Reducir TF-IDF con SVD (representación densa)

SVD combina las características originales en un vector de 100 dimensiones por
usuario y después lo normaliza con L2. Usa las matrices ya generadas: no vuelve
a leer ni preprocesar los textos. Es una reducción de dimensionalidad, no un
embedding contextual preentrenado. Se ajusta con todos los usuarios.

```powershell
.\.venv\Scripts\python.exe -m vectorizacion.svd --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo
.\.venv\Scripts\python.exe -m vectorizacion.svd --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_completo
```

Esto crea `resultados_tfidf/datos_1_a_600/resultados_tfidf_completo_svd100` y
`resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_completo_svd100`. Los TF-IDF de entrada se conservan.
No hacen falta dependencias adicionales a las de vectorización.

| Archivo de SVD | Contenido |
|---|---|
| `X.npy` | Matriz densa float32, normalizada L2: mismos usuarios y orden. |
| `usuarios.json` | Copia exacta del listado de usuarios de su TF-IDF de origen. |
| `svd.joblib` | Modelo que aplica SVD y normalización L2 juntos. |
| `resumen.json` | Dimensiones, semilla, huellas del origen, varianza explicada y vectores cero. |

Por defecto se usan 100 dimensiones, semilla 42 y 7 iteraciones del algoritmo
aleatorizado de TruncatedSVD. Las 100 dimensiones son un punto de partida para
comparar, no un valor óptimo demostrado. La varianza explicada se calcula antes
de normalizar L2; no es un porcentaje de aciertos ni garantiza mejores grupos.

Para probar otra dimensión, añade `--dimensiones 200`; la carpeta predeterminada
terminará en `_svd200`. El valor debe ser menor que el número de usuarios y que
el número de columnas de entrada (aquí, como máximo 597). Puedes cambiar
`--semilla 42`. Para repetir una ejecución, indica otra carpeta con `--salida`:
no se sobrescriben resultados. Si hay filas cero, se conservan y se identifican
en el resumen para revisarlas antes del clustering.

**Cargar y comparar:**

```python
from vectorizacion.matrices import cargar_matriz

X = cargar_matriz("resultados_tfidf/datos_1_a_600/resultados_tfidf_completo_svd100")  # ndarray, 598 × 100
# La misma función carga X.npz como matriz dispersa.
```

```powershell
.\.venv\Scripts\python.exe -m experimentos --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo_svd100 --salida resultados_experimentos_svd_palabras
.\.venv\Scripts\python.exe -m experimentos --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_completo --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_completo_svd100 --salida resultados_experimentos_svd_caracteres
.\.venv\Scripts\python.exe -m Clustering_jerarquico --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo_svd100
.\.venv\Scripts\python.exe -m unittest vectorizacion.test_svd
```

El banco de experimentos exige los mismos IDs en el mismo orden para comparar.
El remuestreo usa vectores fijos: no reajusta TF-IDF ni SVD dentro de cada muestra.

**Transformar usuarios nuevos:** conserva también la carpeta TF-IDF de origen.
El modelo SVD solo recibe vectores con el vocabulario y orden de columnas de ese
TF-IDF. Usa el mismo preprocesamiento y los dos modelos guardados, sin reajustarlos:

```python
import joblib
from vectorizacion.tfidf import cargar_historiales_csv

tfidf = joblib.load("resultados_tfidf_completo/vectorizador.joblib")
svd = joblib.load("resultados_tfidf_completo_svd100/svd.joblib")
usuarios = cargar_historiales_csv("otros_usuarios_preprocesados.csv")
X_nuevos = svd.transform(tfidf.transform([u.publicaciones for u in usuarios]))
```

## 8. Elegir dimensiones: curva de varianza y umbrales del 90 % y 95 %

Desde la raíz de `MD`, con los TF-IDF completos ya generados:

```powershell
.\.venv\Scripts\python.exe -m pip install -r vectorizacion/requirements.txt
.\.venv\Scripts\python.exe -m vectorizacion.analisis_svd --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_completo
```

Se crea `resultados_tfidf/datos_1_a_600/resultados_analisis_svd` con:

| Archivo | Para qué sirve |
|---|---|
| `INFORME.md` | Resumen legible con los mínimos para el 90 % y el 95 % y la tabla comparativa. |
| `varianza_acumulada.png` / `.svg` | Gráfica de ambas representaciones; SVG permite ampliar sin perder calidad. |
| `comparativa.csv` | Valores de 50, 100, 200, 300, 400 y 500 dimensiones, más los cruces y el máximo. |
| `umbrales.csv` | Primer cruce de cada umbral y varianza de la dimensión anterior. |
| `curva_completa.csv` | Varianza de cada componente y acumulada en todas las dimensiones evaluadas. |
| `resumen.json` | Método, huellas de las entradas, curvas y umbrales. |

Los CSV y el JSON expresan las proporciones entre 0 y 1; el informe y la gráfica
las muestran como porcentajes. Un umbral no alcanzado se indica explícitamente.
Las dimensiones solicitadas fuera de rango se omiten de la tabla y se registran
en el JSON. Se exige la misma lista y orden de usuarios para comparar entradas.

Para repetir, añade `--salida resultados_tfidf/datos_1_a_600/resultados_analisis_svd_v2`. Puedes cambiar los
puntos de la tabla con `--dimensiones 50 100 150 200 300` y los objetivos con
`--umbrales 0.80 0.90 0.95`. La búsqueda del primer cruce siempre recorre todas
las dimensiones de la curva, aunque no aparezcan en esa tabla.

**Cómo se calcula:** se obtiene la SVD sin centrar a partir de la matriz de
similitudes algebraicas entre usuarios `X @ X.T`. Para estos datos, esa matriz
es de 598 × 598. Se calcula la varianza de las proyecciones antes de normalizar
L2. No se convierte todo TF-IDF a formato denso ni se cambia a PCA. La memoria
necesaria crece con el cuadrado del número de usuarios; está pensado para este
conjunto de cientos de usuarios.

Se estudian de 1 a `min(usuarios, características) - 1` dimensiones: aquí 597,
el mismo límite del comando `vectorizacion.svd`. La curva usa diagonalización
en float64. Puede diferir ligeramente del modelo SVD aleatorizado ya guardado
o de un nuevo ajuste; hay que comprobar el `resumen.json` de cada modelo que
finalmente se genere. No se modifica el resultado anterior de 100 dimensiones.

**Cómo usarlo:** el 90 % y el 95 % son referencias de conservación de varianza,
no garantías de buenos grupos. Este comando genera el análisis; no sustituye
matrices ni crea automáticamente modelos en las dimensiones elegidas. Cuando
elijáis una, generadla con `python -m vectorizacion.svd --entrada <carpeta>
--dimensiones <numero>` y comparad el clustering con las otras alternativas.

Para comprobar el análisis:

```powershell
.\.venv\Scripts\python.exe -m unittest vectorizacion.test_analisis_svd
```

## 9. Probar modelos con al menos un 95 % de varianza

Las carpetas `resultados_tfidf/datos_1_a_600/resultados_tfidf_completo_svd95pct` y
`resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_completo_svd95pct` contienen nuevas representaciones
para comparar con los TF-IDF originales y con SVD de 100 dimensiones. `95pct`
indica el objetivo de varianza, no el número de dimensiones. Conservan los mismos
598 usuarios y su orden, con normalización L2 y los cuatro archivos habituales:
`X.npy`, `usuarios.json`, `svd.joblib` y `resumen.json`.

| Representación nueva | Dimensiones | Varianza explicada del modelo |
|---|---:|---:|
| Palabras, objetivo 95 % | 495 | 95,04 % |
| Caracteres de longitud 4, objetivo 95 % | 479 | 95,04 % |

Para regenerarlas con estos mismos TF-IDF y la semilla 42:

```powershell
.\.venv\Scripts\python.exe -m vectorizacion.svd --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo --dimensiones 495 --salida resultados_tfidf/datos_1_a_600/resultados_tfidf_completo_svd95pct
.\.venv\Scripts\python.exe -m vectorizacion.svd --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_completo --dimensiones 479 --salida resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_completo_svd95pct
```

Si las carpetas ya existen, usa otro nombre con `--salida`. Estos números de
dimensiones se han elegido para los datos actuales; con otro corpus, repite el
análisis y comprueba que `varianza_explicada_acumulada` del modelo alcance 0.95.
Los mínimos de la curva de diagonalización no garantizan el mismo porcentaje
con SVD aleatorizada: por ejemplo, el ajuste de palabras con 494 dimensiones
dio 94,98 %, por lo que se aumentó a 495.

Para comparar cada nueva representación con TF-IDF y SVD de 100 dimensiones:

```powershell
.\.venv\Scripts\python.exe -m experimentos --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo_svd100 --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_completo_svd95pct --salida resultados_experimentos_comparacion_palabras_95pct
.\.venv\Scripts\python.exe -m experimentos --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_completo --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_completo_svd100 --entrada resultados_tfidf/datos_1_a_600/resultados_tfidf_caracteres4_completo_svd95pct --salida resultados_experimentos_comparacion_caracteres_95pct
```

Puedes añadir `--estabilidad 5` para incluir el submuestreo con vectores fijos.
Hay que evaluar los resultados del clustering: conservar más varianza no
garantiza mejores grupos. La generación de estas carpetas no ejecuta esa comparación.

## Resultados generados el 19/10/2026

Con `data_1_to_600` y el preprocesador actual (idioma automático, mínimo 3 tokens,
máximo 1.000), las 291.787 publicaciones originales quedan en 217.571.
Se conservan 598 usuarios: `subject2538` y `subject2569` quedan sin texto útil
durante el preprocesado. La vectorización conserva los 598, sin vectores cero.

| Representación | Filas (usuarios) | Columnas |
|---|---:|---:|
| Palabras | 598 | 29.068 |
| Caracteres de longitud 4 | 598 | 67.933 |

Las reducciones SVD generadas sobre esas matrices conservan los mismos 598
usuarios, sin vectores cero. Cada `X.npy` ocupa aproximadamente 234 KiB:

| Representación reducida | Dimensiones | Varianza explicada antes de L2 |
|---|---:|---:|
| Palabras + SVD | 100 | 40,67 % |
| Caracteres de longitud 4 + SVD | 100 | 42,77 % |

El análisis completo de la curva obtiene estos mínimos:

| Representación | Para alcanzar 90 % | Para alcanzar 95 % |
|---|---:|---:|
| Palabras | 430 dimensiones | 494 dimensiones |
| Caracteres de longitud 4 | 411 dimensiones | 478 dimensiones |

En la curva calculada por diagonalización, las primeras 100 dimensiones explican
41,00 % y 43,08 %, respectivamente. La pequeña diferencia con los modelos de la
tabla anterior se debe al cálculo aproximado de su SVD aleatorizada. Los modelos
anteriores se conservan; estos mínimos corresponden a la curva del análisis.

En esta ejecución, el preprocesador usa un diccionario de abreviaturas vacío:
busca `./chat_words.json` en la raíz, aunque el archivo está en `preprocesador/`.
Se ha mantenido ese comportamiento del preprocesador. Si se cambia, habrá que
regenerar el CSV y ambas representaciones.
