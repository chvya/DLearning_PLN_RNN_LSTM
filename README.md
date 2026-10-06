# Deep learning en PLN - RNN y LSTM

**ESPOCH · Facultad de Informática y Electrónica · Ingeniería en Software**  
**Base de Conocimiento · Séptimo semestre · Grupo 5**

Este proyecto aplica redes neuronales recurrentes (RNN) y redes de memoria a largo y corto plazo (LSTM) para predecir la siguiente palabra de un texto en español. La práctica permite comparar cómo procesan una secuencia y qué resultados obtienen con los mismos datos.

**Integrantes:** Angel David Gadvay Cepeda, Walter Padilla, Jaime Morales, Jose Nieto, Bryan Lema y Max Viteri.

La [presentación de la investigación](presentacion/Presentacion_RNN_LSTM.pdf) contiene 12 diapositivas. El informe se entrega por separado en el aula virtual.

## Contenido del repositorio

| Carpeta o archivo | Para qué sirve |
|---|---|
| `codigo/` | Preparar datos, entrenar las redes, evaluar sus resultados y ejecutar el autocompletado. Incluye las pruebas del programa. |
| `datos/originales/` | Corpus utilizado, con su atribución y licencia. |
| `datos/procesados/` | Vocabulario, separación de documentos y ejemplos que recibe el programa. |
| `modelos/` | Diez modelos entrenados, listos para consultar y comparar. |
| `resultados/` | Métricas, registros por época, ejemplos de predicción y gráficas. |
| `presentacion/` | Diapositivas en PDF. |
| `requirements-lock.txt` | Versiones de las bibliotecas del programa. |

Los datos y los modelos se incluyen para que la demostración funcione sin entrenar de nuevo. Los registros permiten comprobar las cifras de la investigación.

## Descargar e instalar

En GitHub, selecciona **Code → Download ZIP** y extrae el archivo. Si utilizas Git, puedes descargarlo así:

```powershell
git clone https://github.com/chvya/DLearning_PLN_RNN_LSTM.git
cd DLearning_PLN_RNN_LSTM
```

Abre PowerShell dentro de la carpeta del proyecto. Se utilizó **Python 3.12 de 64 bits en Windows**, con PyTorch para CPU; no hace falta una tarjeta gráfica dedicada.

Comprueba que Python esté instalado:

```powershell
py -3.12 --version
```

Si no aparece la versión, instala Python 3.12 desde [python.org](https://www.python.org/downloads/) antes de continuar. Después ejecuta estas instrucciones, una por una:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

La instalación requiere Internet. Para las siguientes ejecuciones basta con usar el entorno creado; no necesitas instalar las bibliotecas cada vez. Estas instrucciones corresponden a Windows; no se ha verificado una instalación nueva en Linux o macOS.

## Probar el autocompletado

```powershell
.\.venv\Scripts\python.exe codigo/demo.py
```

Cuando aparezca `Contexto>`, escribe una frase y pulsa Enter. Por ejemplo:

```text
el presidente del gobierno
```

El programa carga la LSTM seleccionada mediante validación y muestra cinco palabras posibles. En esta consulta, las sugerencias guardadas son `de`, `del`, `vasco`, `es` y `se`, con probabilidades aproximadas de 13,77 %, 7,35 %, 3,00 %, 1,97 % y 1,97 %.

- Escribe `/salir` para cerrar la demostración.
- El programa predice una palabra después del contexto; no completa letras de una palabra a medio escribir.
- Una sugerencia no garantiza que la oración sea correcta. El vocabulario y los textos de entrenamiento limitan las respuestas.
- Las cinco probabilidades no suman 100 % porque existen otras palabras posibles. La salida también indica la probabilidad de `UNK`, que representa palabras fuera del vocabulario.
- La consulta funciona sin Internet y no vuelve a entrenar el modelo.

Para hacer una sola consulta desde PowerShell:

```powershell
.\.venv\Scripts\python.exe codigo/pln.py predecir --texto "el presidente del gobierno"
```

## Comprobar el programa

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s codigo -p test_*.py -v
```

Se ejecutan ocho pruebas. Comprueban la normalización del texto, la separación de documentos, la construcción del contexto sin palabras futuras, el relleno de secuencias, el tratamiento de palabras desconocidas, la distribución del bigrama y la conservación del modelo al guardarlo y cargarlo. Al finalizar correctamente, la terminal muestra `OK`.

| Dificultad | Revisión recomendada |
|---|---|
| No se encuentra `codigo/demo.py` | Abre PowerShell en la carpeta principal del proyecto. |
| No se encuentra `.venv` o falta `torch` | Completa la instalación y utiliza el ejecutable de Python indicado arriba. |
| No hay LSTM entrenada | Comprueba que descargaste completas las carpetas `modelos/` y `resultados/`. |
| Aparecen palabras desconocidas | Es una limitación del vocabulario. Prueba la consulta de referencia para comprobar el funcionamiento. |

## Datos y configuración del experimento

Se utilizó [UD Spanish-AnCora, versión r2.17](https://github.com/UniversalDependencies/UD_Spanish-AnCora/tree/r2.17), que contiene textos periodísticos en español. Los textos se pasan a minúsculas, conservan las tildes, sustituyen números por `<num>` y omiten puntuación. Se descartaron 62 oraciones exactamente repetidas.

La separación es propia del experimento: se agrupan las oraciones por documento y se asigna cada documento a entrenamiento, validación o prueba mediante una regla determinista basada en SHA-256. No corresponde a las particiones oficiales de evaluación de UD. El vocabulario se construye únicamente con entrenamiento.

| Elemento | Configuración |
|---|---|
| Vocabulario | 3 000 entradas, incluidos `PAD`, `UNK` y `BOS` |
| Objetivos de entrenamiento | 100 000 seleccionados |
| Objetivos de validación | 52 695 |
| Objetivos de prueba | 56 796 |
| Contexto de la comparación principal | Hasta 12 posiciones anteriores, incluida la señal de inicio cuando corresponda |
| Representación de cada palabra | 48 valores |
| Estado oculto | 64 valores |
| Entrenamiento | 5 épocas, lotes de 512, Adam con tasa 0,003 |
| Regularización | Dropout 0,15 y recorte de gradiente a 1,0 |
| Semillas principales | 17, 29 y 43 para cada arquitectura |
| Comparación adicional | Contextos de 4 y 24 posiciones, con semilla 17 |

Las seis ejecuciones principales y las cuatro adicionales suman diez modelos. El piloto inicial no se incluye en este repositorio ni en las medias finales. El mejor punto de cada entrenamiento se conserva según su pérdida de validación; el conjunto de prueba no se utiliza para elegirlo.

## Resultados

| Modelo | Perplejidad de prueba | Acierto entre las primeras cinco palabras |
|---|---:|---:|
| Bigrama de referencia | 85,93 | 25,62 % |
| RNN, media de tres semillas | 75,60 | 26,58 % |
| LSTM, media de tres semillas | 78,15 | 26,14 % |

Una perplejidad menor indica que el modelo asignó mejor probabilidad a las respuestas evaluadas; no es un porcentaje de acierto. El acierto top-5 cuenta si la palabra correcta aparece entre las cinco primeras candidatas. `UNK` interviene en la pérdida, pero no cuenta como palabra acertada.

La RNN obtuvo mejores medias en esta configuración. Esto no demuestra que siempre sea mejor que LSTM. Además, el 21,19 % de los objetivos de prueba quedó fuera del vocabulario, lo que limita la cobertura. La demo utiliza la mejor LSTM según validación para mostrar el manejo de memoria estudiado en la actividad.

Para verificar los datos:

- `resultados/evaluacion_test.json`: resultados de cada modelo y del bigrama.
- `resultados/resumen_comparacion.json`: medias y desviaciones de las tres semillas principales.
- `resultados/metricas.csv`: tabla de resultados para consultar en una hoja de cálculo.
- `resultados/*_historial.json`: evolución del entrenamiento por época.
- `resultados/ejemplos_demo.json`: consultas y respuestas guardadas.

## Repetir el experimento

Para preparar los datos, ejecutar las pruebas, entrenar las diez configuraciones y generar los resultados en una carpeta nueva:

```powershell
.\.venv\Scripts\python.exe codigo/reproducir.py --destino replicas/replica_01
```

El destino no debe existir. La réplica conserva los archivos originales y tarda más que una consulta. No hace falta ejecutarla antes de cada demostración.

Si solo necesitas reconstruir las tablas, gráficas y ejemplos desde los resultados existentes:

```powershell
.\.venv\Scripts\python.exe codigo/analizar.py
```

Esta orden actualiza los archivos derivados de `resultados/` y guarda las gráficas en `resultados/figuras/`. No entrena las redes. Las versiones originales de los pesos y los registros de entrenamiento se conservan.

## Organización del código

| Archivo | Función |
|---|---|
| `pln.py` | Normalización, preparación, redes RNN/LSTM, entrenamiento, bigrama, evaluación y predicción. |
| `demo.py` | Consulta interactiva desde la terminal. |
| `experimentos.py` | Ejecución de las diez configuraciones establecidas. |
| `analizar.py` | Resumen de métricas y generación de tablas y figuras. |
| `test_pln.py` | Ocho pruebas de integridad. |
| `reproducir.py` | Repetición del procedimiento en una carpeta nueva. |

## Atribución

El corpus se atribuye a Taulé, Martí y Recasens y a los contribuidores de UD Spanish-AnCora. Su licencia **CC BY 4.0** se conserva en `datos/originales/LICENSE.txt`, junto con la documentación original. Los datos procesados incorporan las transformaciones descritas arriba.
