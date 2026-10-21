# Resultados de la vectorización TF-IDF

Todas las representaciones TF-IDF (y sus reducciones SVD) están aquí, separadas
según los datos de origen. Los XML y los CSV **no** están en el repositorio.

```text
resultados_tfidf/
├── datos_1_a_600/      solo la carpeta data_1_to_600 (598 usuarios útiles)
└── datos_completos/    las cinco carpetas: data_1_to_600 … data_2001_to_2348 (2.344 usuarios útiles)
```

## datos_1_a_600

Los resultados anteriores, sin cambios (solo se han movido de la raíz a esta
carpeta). Diez representaciones y el análisis de varianza:

| Carpeta | Usuarios | Dimensiones |
|---|---:|---:|
| `resultados_tfidf` (histórico, train/test) | 480 + 120 | 37.246 |
| `resultados_tfidf_caracteres4` (histórico, train/test) | 480 + 120 | 138.112 |
| `resultados_tfidf_historico_alineado` | 598 | 37.246 |
| `resultados_tfidf_caracteres4_historico_alineado` | 598 | 138.112 |
| `resultados_tfidf_completo` | 598 | 29.068 |
| `resultados_tfidf_caracteres4_completo` | 598 | 67.933 |
| `resultados_tfidf_completo_svd100` | 598 | 100 |
| `resultados_tfidf_caracteres4_completo_svd100` | 598 | 100 |
| `resultados_tfidf_completo_svd95pct` | 598 | 495 |
| `resultados_tfidf_caracteres4_completo_svd95pct` | 598 | 479 |
| `resultados_analisis_svd` | — | curva de varianza |

## datos_completos

Mismo procedimiento que el modo completo de los 600, aplicado a los 2.348
usuarios de las cinco carpetas. Los históricos con train/test **no** se
repiten: eran la primera versión del proceso y solo existen para los 600.

Preprocesado (21/10/2026): idioma automático, mínimo 3 tokens, máximo 1.000 y
diccionario de abreviaturas vacío, igual que en la ejecución de los 600. Las
1.018.484 publicaciones originales quedan en 836.472. Cuatro usuarios se quedan
sin texto útil: `subject2538`, `subject2569`, `subject5035` y `subject741`.
Ninguna representación tiene vectores cero.

| Carpeta | Usuarios | Dimensiones | Varianza (antes de L2) |
|---|---:|---:|---:|
| `resultados_tfidf_completo` | 2.344 | 64.617 | — |
| `resultados_tfidf_caracteres4_completo` | 2.344 | 109.148 | — |
| `resultados_tfidf_completo_svd100` | 2.344 | 100 | 28,80 % |
| `resultados_tfidf_caracteres4_completo_svd100` | 2.344 | 100 | 30,63 % |
| `resultados_tfidf_completo_svd95pct` | 2.344 | 1.821 | 95,01 % |
| `resultados_tfidf_caracteres4_completo_svd95pct` | 2.344 | 1.747 | 95,00 % |
| `resultados_analisis_svd` | — | — | curva de varianza |

Las seis tienen los mismos 2.344 usuarios en el mismo orden. Con más usuarios,
100 dimensiones conservan menos varianza que con 598, y el 95 % exige
unas 1.800 dimensiones en lugar de unas 490. En la curva de diagonalización el
95 % se cruza en 1.816 (palabras) y 1.743 (caracteres); la SVD aleatorizada
necesitó 1.821 y 1.747 para superar 0,95 en su propio resumen.

**Modelos que no están en GitHub:** los dos `svd.joblib` de `*_svd95pct`
ocupan 424 MB y 686 MB: superan el límite de 100 MB por archivo de GitHub y,
juntos, la cuota gratuita de 1 GB de Git LFS. Se quedan en local (están en
`.gitignore`). Las matrices `X.npy`, `usuarios.json` y `resumen.json` sí están.

**¿Los necesitas?** Para los experimentos de clustering, no: `experimentos`,
`bootstrap` y `Clustering_jerarquico` solo leen `X.npy`/`X.npz` y
`usuarios.json`. Solo hacen falta para clasificar textos nuevos
(`experimentos.consultar`) con una representación SVD95pct de
`datos_completos`. Para generarlos sin rehacer nada más (varios minutos cada uno):

```powershell
$s = "resultados_tfidf\datos_completos"
.\.venv\Scripts\python.exe -m vectorizacion.svd --entrada "$s\resultados_tfidf_completo" --dimensiones 1821 --salida "$s\resultados_tfidf_completo_svd95pct_regenerado"
.\.venv\Scripts\python.exe -m vectorizacion.svd --entrada "$s\resultados_tfidf_caracteres4_completo" --dimensiones 1747 --salida "$s\resultados_tfidf_caracteres4_completo_svd95pct_regenerado"
Copy-Item "$s\resultados_tfidf_completo_svd95pct_regenerado\svd.joblib" "$s\resultados_tfidf_completo_svd95pct\"
Copy-Item "$s\resultados_tfidf_caracteres4_completo_svd95pct_regenerado\svd.joblib" "$s\resultados_tfidf_caracteres4_completo_svd95pct\"
```

Se usa una carpeta intermedia porque `vectorizacion.svd` no sobrescribe
carpetas con contenido; después puedes borrar las `*_regenerado`. Con la misma
semilla (42) y el mismo TF-IDF se obtiene el mismo modelo, salvo diferencias
numéricas mínimas si cambia la versión de scikit-learn (se generó con la 1.9.1).
Comprueba que `varianza_explicada_acumulada` del nuevo `resumen.json` sea ≥ 0,95.

## Regenerar datos_completos

Desde la raíz de `MD`. El preprocesador siempre escribe en `output/`, así que
aparta antes los CSV de los 600 si quieres conservarlos:

```powershell
.\.venv\Scripts\python.exe -m preprocesador --datos "C:\ruta\a\la\carpeta\con\las\cinco\data_*" --guardar_csv
# mover output\data.csv y output\data_preprocesado.csv a output\datos_completos\

$c = "output\datos_completos\data_preprocesado.csv"
$s = "resultados_tfidf\datos_completos"
.\.venv\Scripts\python.exe -m vectorizacion --datos $c --representacion palabras --salida "$s\resultados_tfidf_completo"
.\.venv\Scripts\python.exe -m vectorizacion --datos $c --representacion caracteres4 --salida "$s\resultados_tfidf_caracteres4_completo"
.\.venv\Scripts\python.exe -m vectorizacion.svd --entrada "$s\resultados_tfidf_completo"
.\.venv\Scripts\python.exe -m vectorizacion.svd --entrada "$s\resultados_tfidf_caracteres4_completo"
.\.venv\Scripts\python.exe -m vectorizacion.analisis_svd --entrada "$s\resultados_tfidf_completo" --entrada "$s\resultados_tfidf_caracteres4_completo" --dimensiones 50 100 200 300 400 500 750 1000 1500 2000 --salida "$s\resultados_analisis_svd"
.\.venv\Scripts\python.exe -m vectorizacion.svd --entrada "$s\resultados_tfidf_completo" --dimensiones 1821 --salida "$s\resultados_tfidf_completo_svd95pct"
.\.venv\Scripts\python.exe -m vectorizacion.svd --entrada "$s\resultados_tfidf_caracteres4_completo" --dimensiones 1747 --salida "$s\resultados_tfidf_caracteres4_completo_svd95pct"
```

Los SVD de ~1.800 dimensiones tardan varios minutos cada uno.

## Usar uno u otro conjunto

```powershell
.\.venv\Scripts\python.exe -m experimentos.todas --comprobar-entradas                              # 600 (8 entradas)
.\.venv\Scripts\python.exe -m experimentos.todas --conjunto datos_completos --comprobar-entradas   # completos (6 entradas)
.\.venv\Scripts\python.exe -m Clustering_jerarquico --entrada resultados_tfidf\datos_completos\resultados_tfidf_completo_svd100 --salida resultados_clustering_datos_completos
```

Para `experimentos.consultar` con los datos completos, indica también sus textos:
`--textos output\datos_completos\data_preprocesado.csv` (por defecto lee los de los 600).

No mezcles carpetas de los dos conjuntos en una misma comparación: tienen
usuarios distintos y el banco de pruebas la rechazará.
