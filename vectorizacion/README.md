# Cómo ejecutar la vectorización

Este programa lee los XML de CLEF y convierte el historial de cada usuario en
un vector numérico. Hay dos opciones: **TF-IDF de palabras** y **TF-IDF de
caracteres de longitud 4**. Los XML originales no se modifican.

## 1. Preparar la carpeta

Necesitas Python 3.10 o superior instalado, los XML y la carpeta completa
`vectorizacion`. La estructura debe ser así:

```text
data_1_to_600/
├── subject1.xml
├── subject100.xml
├── ...
└── vectorizacion/
    ├── __init__.py
    ├── __main__.py
    ├── tfidf.py
    ├── test_tfidf.py
    ├── requirements.txt
    └── README.md
```

Abre PowerShell en **data_1_to_600**, la carpeta que contiene los XML y
`vectorizacion`. Ejecuta todos los comandos siguientes desde ahí.

## 2. Instalar las dependencias (solo la primera vez)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r vectorizacion/requirements.txt
```

Si `python` no se reconoce y tienes el lanzador de Python instalado, puedes usar
`py -m venv .venv` en el primer comando. No hace falta activar el entorno virtual.
Si `.venv` ya existe, puedes usarlo e instalar las dependencias con el segundo comando.

## 3. Ejecutar la opción que necesites

**TF-IDF de palabras:**

```powershell
.\.venv\Scripts\python.exe -m vectorizacion --representacion palabras --salida resultados_palabras_v2
```

**TF-IDF de caracteres de longitud 4:**

```powershell
.\.venv\Scripts\python.exe -m vectorizacion --representacion caracteres4 --salida resultados_caracteres4_v2
```

Por ejemplo, la segunda opción convierte `gambling` en los fragmentos `gamb`,
`ambl`, `mbli`, `blin` y `ling` antes de calcular sus pesos.

**Para repetir una ejecución, cambia el nombre de `--salida`**, por ejemplo a
`resultados_palabras_v3`. El programa exige una carpeta nueva o vacía para evitar
sobrescribir resultados anteriores.

Si los XML están en otra carpeta, añade `--datos "C:\ruta\a\los\xml"` al comando.
Sigue ejecutándolo desde la carpeta que contiene el paquete `vectorizacion`.

## 4. Recoger los resultados

Cada carpeta de salida contiene:

| Archivo | Qué contiene |
|---|---|
| `X_train.npz` | Vectores de los usuarios de entrenamiento. |
| `X_test.npz` | Vectores de los usuarios reservados para prueba. |
| `usuarios_train.json` y `usuarios_test.json` | Usuario y XML que corresponde a cada fila. |
| `vectorizador.joblib` | Vocabulario y pesos aprendidos, para transformar textos nuevos. |
| `resumen.json` | Parámetros, dimensiones y posibles usuarios con vector vacío. |

**Para compartir el trabajo, entrega la carpeta de resultados y la carpeta
`vectorizacion`.** Tus compañeros también necesitarán los XML si quieren consultar
los textos originales. El entorno `.venv` se crea en cada equipo; no hace falta enviarlo.

## Qué se hace por defecto

- Se genera **una fila por usuario**, usando los títulos y textos de sus publicaciones.
- Se usa el **80% de usuarios para entrenamiento y el 20% para prueba**, con semilla 42.
- El vocabulario y los pesos IDF se aprenden solo con entrenamiento.
- Se conservan atributos presentes en al menos 3 usuarios de entrenamiento.
- Los vectores se normalizan con L2 y se guardan como matrices dispersas para ahorrar memoria.

Con los mismos XML y parámetros, ambas opciones usan los mismos usuarios en el
mismo orden. Con los 600 usuarios actuales son **480 de entrenamiento y 120 de prueba**.
La ejecución comprobada con scikit-learn 1.8.0 produjo 37.246 atributos de palabras
y 138.112 de caracteres. Si cambian los datos o parámetros, estas cantidades pueden variar.

## Comprobar la instalación o consultar opciones

```powershell
.\.venv\Scripts\python.exe -m unittest vectorizacion.test_tfidf
.\.venv\Scripts\python.exe -m vectorizacion --help
```

Si aparece `No module named vectorizacion`, comprueba que estás en la carpeta
que contiene `vectorizacion`, no dentro de ella. Si falta `sklearn`, `scipy` u
otra dependencia, repite el comando de instalación del paso 2.
