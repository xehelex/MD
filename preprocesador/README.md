
# Cómo ejecutar el preprocesador y el análisis exploratorio

Este programa lee los XML de CLEF y extrae las publicaciones de cada usuario.
Con ellas puede hacer dos cosas, de forma independiente:

- **Preprocesar** los textos para tareas de clustering: eliminación de HTML y URLs,
  conversión de jerga, normalización, tokenización, eliminación de stopwords y
  lematización de textos en inglés.
- **Analizar** el dataset (`--analisis`): valores ausentes, longitudes, duplicados y
  distribución de idiomas (inglés, español y otros).

Los XML originales no se modifican.

## 1. Preparar la carpeta

Necesitas Python 3.11 o superior instalado, los XML de CLEF y
la carpeta completa `preprocesador`.

La estructura debe ser así:

```text
data_1_to_600/
├── subject1.xml
├── subject100.xml
├── ...
├── preprocesador/
│   ├── __init__.py
│   ├── __main__.py
│   ├── preprocesador.py
│   ├── chat_words.json
│   ├── requirements.txt
│   ├── test_preprocesador.py
│   └── README.md
└── output/
```

El archivo `chat_words.json` contiene el diccionario de jerga utilizado durante
la limpieza. Si no existe, se utilizará un diccionario vacío.

Abre PowerShell en `data_1_to_600`, la carpeta que contiene los XML
y `preprocesador`. Ejecuta todos los comandos siguientes desde ahí.

## 2. Instalar las dependencias (solo la primera vez)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r preprocesador/requirements.txt
```

Si `python` no se reconoce y tienes el lanzador de Python instalado,
puedes usar `py -m venv .venv` en el primer comando.

No hace falta activar el entorno virtual.

Si `.venv` ya existe, puedes utilizarlo e instalar las dependencias
con el segundo comando.

La primera ejecución puede necesitar conexión a Internet para
descargar los recursos de NLTK utilizados en la tokenización, stopwords
y lematización.

## 3. Ejecutar

El programa busca todos los archivos XML dentro del directorio indicado en
`--datos`, incluyendo sus subcarpetas. La ejecución debe realizarse desde la
carpeta que contiene el paquete `preprocesador`, no desde dentro de la propia carpeta.

### 3.1. Preprocesar los datos

```powershell
.\.venv\Scripts\python.exe -m preprocesador --datos "." --guardar_csv
```

Sin `--guardar_csv` el programa procesa los datos pero no escribe ningún archivo.

Si los XML están en otra carpeta, indica su ruta:

```powershell
.\.venv\Scripts\python.exe -m preprocesador --datos "C:\ruta\a\los\xml" --guardar_csv
```

Ejemplo con todas las opciones de preprocesamiento:

```powershell
.\.venv\Scripts\python.exe -m preprocesador --datos "." --idioma english --min_tokens 5 --max_tokens 500 --guardar_csv
```

### 3.2. Análisis exploratorio

```powershell
.\.venv\Scripts\python.exe -m preprocesador --datos "." --analisis
```

Para guardar además el gráfico de idiomas:

```powershell
.\.venv\Scripts\python.exe -m preprocesador --datos "." --analisis --guardar_fig output/idiomas.png
```

Con `--analisis` solo se ejecuta el análisis: no se preprocesan los textos ni se
generan los CSV. Los resultados no se imprimen por pantalla, sino que se escriben
en `output/analisis.txt`. Además, se abre una ventana con el gráfico de
distribución de idiomas y el programa finaliza al cerrarla.

### 3.3. Opciones

| Opción | Descripción | Valor por defecto |
|---|---|---|
| `--datos` | Carpeta con los XML (obligatoria). | |
| `--analisis` | Ejecuta únicamente el análisis exploratorio. | Desactivada |
| `--guardar_fig` | Con `--analisis`: ruta donde guardar el gráfico de idiomas. | No se guarda |
| `--guardar_csv` | Guarda `data.csv` y `data_preprocesado.csv` en `output`. | Desactivada |
| `--idioma` | `auto`, `english` o `spanish`. | `auto` |
| `--min_tokens` | Se descartan las publicaciones con menos tokens tras la limpieza. | `3` |
| `--max_tokens` | Se truncan las publicaciones con más tokens. | `1000` |

## 4. Recoger los resultados

El programa genera la carpeta `output` si no existe.

Dentro se guardan los siguientes archivos:

| Archivo | Se genera con | Qué contiene |
|---|---|---|
| `data.csv` | `--guardar_csv` | Publicaciones extraídas de los XML, con sus metadatos originales. |
| `data_preprocesado.csv` | `--guardar_csv` | Publicaciones con la columna `text_Clustering`, que contiene los textos preprocesados. |
| `analisis.txt` | `--analisis` | Informe del análisis exploratorio. |
| `idiomas.png` | `--analisis --guardar_fig output/idiomas.png` | Gráfico de distribución de idiomas. |

La columna `text_Clustering` contiene los tokens de cada publicación separados
por espacios. Las publicaciones sin texto útil (vacías o con menos tokens que
`--min_tokens`) y los duplicados de un mismo usuario se eliminan.

El archivo `analisis.txt` incluye:

- Número de publicaciones, número de sujetos y rango de fechas.
- Valores ausentes o vacíos por columna.
- Longitud de las publicaciones (caracteres y palabras): mínimo, percentil 5, mediana, media, percentil 95 y máximo.
- Número de textos duplicados exactos.
- Distribución de idiomas (inglés, español y otros) y los códigos de idioma más frecuentes dentro de "Otro". El código `und` indica que no se pudo detectar el idioma.

Los XML originales no se modifican.

**Para compartir el trabajo, entrega la carpeta de resultados y
la carpeta `preprocesador`.** Tus compañeros también necesitarán los
XML originales si quieren volver a ejecutar el preprocesamiento. 
El entorno `.venv` se crea en cada equipo; no hace falta enviarlo.

## Qué se hace por defecto en el preprocesamiento

- Se extraen las publicaciones de los XML, junto con el identificador del usuario, título, fecha, texto e información adicional.
- Se eliminan las URLs y las etiquetas HTML.
- Se convierte la jerga utilizando `chat_words.json`.
- Se detecta el idioma de cada texto. Si se detecta español, se utilizan stopwords españolas. El resto de idiomas se procesa como inglés.
- Se normaliza el texto a minúsculas y se eliminan los signos de puntuación, excepto los apóstrofes y los guiones.
- Se tokeniza el texto y se eliminan las stopwords.
- Se lematizan los textos en inglés mediante POS tagging y WordNet.
- Los textos en español no se lematizan: se reducen a su raíz mediante stemming.

La columna original `text` se conserva y el resultado del preprocesamiento
se añade en `text_Clustering`.

## Comprobar la instalación o consultar opciones

Para ejecutar las pruebas:

```powershell
.\.venv\Scripts\python.exe -m pytest -v preprocesador/test_preprocesador.py
```

Para consultar las opciones disponibles:

```powershell
.\.venv\Scripts\python.exe -m preprocesador --help
```

Si aparece `No module named preprocesador`, comprueba que estás en la carpeta
que contiene el paquete `preprocesador`, no dentro de ella.

Si falta alguna dependencia, repite el comando de instalación del paso 2.
