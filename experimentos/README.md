# Banco de pruebas

Compara configuraciones de clustering jerárquico sobre una o varias matrices ya
vectorizadas y saca una tabla con la calidad de cada partición. Solo lee los
`X_train.npz` que le indiques; no recalcula nada.

## Ejecutar

Desde la carpeta `MD`:

```powershell
python -m pip install -r experimentos/requirements.txt
python -m experimentos --entrada resultados_tfidf
```

Repite `--entrada` para comparar varias matrices:

```powershell
python -m experimentos --entrada resultados_tfidf --entrada resultados_tfidf_limpio
```

Con 480 usuarios tarda unos 15 s por matriz, o un minuto si pides estabilidad.

## Opciones

| Opción | Por defecto | |
|---|---|---|
| `--entrada` | obligatoria | Carpeta con `X_train.npz`. Repetible. |
| `--salida` | `resultados_experimentos` | Dónde dejar la tabla. |
| `--k-min` / `--k-max` | 2 y 20 | Rango de k. |
| `--minimo` | 10 | Tamaño a partir del cual un grupo cuenta. |
| `--estabilidad` | 0 | Repeticiones de remuestreo. Multiplica el tiempo por N. |

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
