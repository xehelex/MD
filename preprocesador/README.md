
# Cómo ejecutar el preprocesador

Este programa lee los XML de CLEF y extrae las publicaciones de cada usuario. 
Después limpia y transforma los textos para prepararlos para tareas de 
clustering.

El preprocesamiento incluye eliminación de HTML y URLs, conversión de jerga, 
normalización, tokenización, eliminación de stopwords y lematización de textos 
en inglés.

Los XML originales no se modifican.

## 1. Preparar la carpeta

Necesitas Python 3.10 o superior instalado, los XML de CLEF y 
la carpeta completa `preprocesador`.

La estructura debe ser así:

```text
data_1_to_600/
├── subject1.xml
├── subject100.xml
├── ...
├── config/
│   └── chat_words.json
├── preprocesador/
│   ├── __init__.py
│   ├── __main__.py
│   ├── preprocesador.py
│   ├── requirements.txt
│   ├── test_preprocesador.py
│   └── README.md
└── output/
```

La carpeta `config` contiene el diccionario de jerga utilizado durante 
la limpieza. Si no existe `chat_words.json`, se utilizará un diccionario vacío.

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

## 3. Ejecutar el preprocesador

Para procesar los XML que están en la carpeta actual:

```powershell
.\.venv\Scripts\python.exe -m preprocesador --datos "."
```

Si los XML están en otra carpeta, añade la ruta correspondiente:

```powershell
.\.venv\Scripts\python.exe -m preprocesador --datos "C:\ruta\a\los\xml"
```

El programa busca todos los archivos XML dentro del directorio indicado, 
incluyendo sus subcarpetas.

La ejecución debe realizarse desde la carpeta que contiene el paquete 
`preprocesador`, no desde dentro de la propia carpeta.

## 4. Recoger los resultados

El programa genera la carpeta `output` si no existe.

Dentro se guardan los siguientes archivos:

| Archivo | Qué contiene |
|---|---|
| `data.csv` | Publicaciones extraídas de los XML, con sus metadatos originales. |
| `data_prepro.csv` | Publicaciones con la columna `text_Clustering` que contiene los textos preprocesados. |

La columna `text_Clustering` contiene listas de tokens. Los textos vacíos 
o nulos se representan mediante una lista vacía.

Los XML originales no se modifican.

**Para compartir el trabajo, entrega la carpeta de resultados y 
la carpeta `preprocesador`.** Tus compañeros también necesitarán los 
XML originales si quieren volver a ejecutar el preprocesamiento. 
El entorno `.venv` se crea en cada equipo; no hace falta enviarlo.

## Qué se hace por defecto

- Se extraen las publicaciones de los XML, junto con el identificador del usuario, título, fecha, texto e información adicional.
- Se eliminan las URLs y las etiquetas HTML.
- Se convierte la jerga utilizando `chat_words.json`.
- Se detecta el idioma de cada texto. Si se detecta español, se utilizan stopwords españolas. El resto de idiomas se procesa como inglés.
- Se normaliza el texto a minúsculas y se eliminan los signos de puntuación, excepto los apóstrofes y los guiones.
- Se tokeniza el texto y se eliminan las stopwords.
- Se lematizan los textos en inglés mediante POS tagging y WordNet.
- Los textos en español no se lematizan.

La columna original `text` se conserva y el resultado del preprocesamiento 
se añade en `text_Clustering`.

## Comprobar la instalación o consultar opciones

Para ejecutar las pruebas:

```powershell
.\.venv\Scripts\python.exe -m pytest -v test_preprocesador.py
```

Para consultar las opciones disponibles:

```powershell
.\.venv\Scripts\python.exe -m preprocesador --help
```

Si aparece `No module named preprocesador`, comprueba que estás en la carpeta 
que contiene el paquete `preprocesador`, no dentro de ella.

Si falta alguna dependencia, repite el comando de instalación del paso 2.
