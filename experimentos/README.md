# Cómo ejecutar el banco de pruebas

Este programa compara configuraciones de clustering jerárquico sobre una o
varias matrices ya vectorizadas y produce una tabla con la calidad de cada
partición. Sirve para responder con números a qué enlace, qué distancia y qué
número de grupos funcionan mejor con nuestros datos.

No recalcula vectorizaciones ni modifica nada: solo lee los `X_train.npz` que
le indiques.

## 1. Qué necesitas

Python 3.10 o superior, el paquete `Clustering_jerarquico` (de donde salen el
algoritmo y las métricas) y al menos una carpeta de resultados de vectorización.
Todo se ejecuta desde la raíz del proyecto, la carpeta `MD`.

```powershell
python -m pip install -r experimentos/requirements.txt
```

## 2. Ejecutar

Sobre una sola matriz:

```powershell
python -m experimentos --entrada resultados_tfidf
```

Comparando varias, repitiendo `--entrada`:

```powershell
python -m experimentos --entrada resultados_tfidf --entrada resultados_tfidf_limpio_sinvacias
```

Con los 480 usuarios de entrenamiento tarda alrededor de medio minuto por cada
dos matrices.

## 3. Opciones

| Opción | Por defecto | Para qué |
|---|---|---|
| `--entrada` | obligatoria | Carpeta con `X_train.npz`. Se puede repetir. |
| `--salida` | `resultados_experimentos` | Dónde dejar la tabla. |
| `--k-min` / `--k-max` | 2 y 20 | Rango de número de grupos a barrer. |
| `--minimo` | 10 | A partir de cuántos miembros un grupo cuenta como aprovechable. |

## 4. Qué devuelve

En la carpeta de salida quedan dos archivos:

| Archivo | Contenido |
|---|---|
| `rejilla.csv` | Una fila por cada combinación de matriz, enlace, distancia y k. |
| `resumen.json` | Parámetros usados y la correlación entre silhouette y equilibrio. |

Las columnas de `rejilla.csv`:

| Columna | Significado |
|---|---|
| `silhouette` | Métrica interna clásica. Cuanto mayor, supuestamente mejor. |
| `equilibrio` | Entropía normalizada de los tamaños de grupo: 1 si todos miden igual, cerca de 0 si uno se lo come todo. |
| `mayor_pct` | Porcentaje de instancias que caen en el grupo más grande. |
| `grupos_utiles` | Cuántos grupos superan `--minimo` miembros. |
| `instancias_sueltas` | Cuántas instancias quedan en grupos por debajo de ese mínimo. |
| `tamanos` | Los seis grupos mayores, de más a menos. |
| `cofenetica` | Correlación entre las distancias originales y las alturas del árbol. |
| `segundos_arbol` | Lo que costó construir ese árbol. |

## Por qué hay una columna de equilibrio

Porque el silhouette por sí solo no distingue una partición repartida de otra
degenerada, y de hecho puntúa mejor la segunda. Con `average` y k=4 sobre la
matriz original, 476 de 480 usuarios caen en un mismo grupo y los otros tres
quedan solos: esa configuración obtiene el silhouette más alto de toda la
rejilla. Mirando el equilibrio al lado, 0,041 sobre 1, se ve de inmediato que no
sirve.

El desglose por enlace matiza el asunto. Dentro de un mismo enlace, silhouette y
equilibrio apenas se relacionan, y con `ward` la correlación llega a ser
positiva. El problema aparece al comparar **entre** enlaces: single y average
puntúan alto con equilibrio ínfimo, ward lo contrario. Como elegir modelo es
precisamente comparar entre enlaces, conviene mirar siempre las dos columnas
juntas.

## Qué no hace todavía

- No calcula Davies-Bouldin, que obliga a densificar la matriz y con 36.000
  atributos resulta caro. Habría que medirlo sobre un espacio reducido.
- No mide estabilidad por remuestreo, aunque `Clustering_jerarquico` ya tiene
  con qué hacerlo.

## Comprobar la instalación

```powershell
python -m unittest experimentos.test_banco
python -m experimentos --help
```

Si aparece `No module named experimentos`, comprueba que estás en la carpeta
`MD` y no dentro de `experimentos`.
