# Predicción de la siguiente palabra con RNN y LSTM

Aplicación local del grupo 5 de Base de Conocimiento (ESPOCH). Escribes una frase y ves cinco palabras que podrían seguir. También puedes revisar el corpus, la preparación de los datos, el historial de entrenamiento y los resultados de prueba.

**La forma principal de usar el proyecto es la interfaz web local.** Los modelos ya están entrenados y vienen dentro del repositorio: no tienes que volver a entrenarlos para abrirla.

## Instalación en Windows

Estas instrucciones se comprobaron con **Windows de 64 bits y Python 3.12**. Necesitas Internet solo para descargar el proyecto e instalar las bibliotecas. Después, la aplicación funciona en tu equipo.

1. En GitHub, pulsa **Code → Download ZIP** y extrae el ZIP completo. También puedes usar `git clone https://github.com/chvya/DLearning_PLN_RNN_LSTM.git`.
2. Abre **PowerShell dentro de la carpeta extraída**. Debes ver ahí `Abrir Ancora.vbs`, `requirements-app.txt` y las carpetas `codigo`, `datos`, `modelos` y `resultados`.
3. Comprueba Python:

   ```powershell
   py -3.12 --version
   ```

   Si no aparece Python 3.12, instálalo desde [python.org](https://www.python.org/downloads/) y vuelve a abrir PowerShell. Usa la versión de **64 bits**.

4. Crea el entorno e instala lo necesario para la aplicación. Ejecuta cada línea desde la carpeta del proyecto:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements-app.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```

   Espera a que termine la instalación antes de abrir la interfaz. **No necesitas activar `.venv`** ni cambiar la política de ejecución de PowerShell: los comandos llaman directamente a su Python.

5. Comprueba que los paquetes y los archivos del proyecto se pueden usar:

   ```powershell
   .\.venv\Scripts\python.exe -c "import torch, numpy; print('PyTorch:', torch.__version__, '| NumPy:', numpy.__version__)"
   .\.venv\Scripts\python.exe -m unittest discover -s codigo -p test_*.py -v
   ```

   Al final de las pruebas debe aparecer `Ran 15 tests` y `OK`. Algunas pruebas abren un servidor **solo en tu equipo** y lo cierran al terminar.

## Abrir y cerrar la interfaz

Haz doble clic en **`Abrir Ancora.vbs`**. El navegador debe abrir `http://127.0.0.1:8765`. La primera carga puede tardar unos segundos mientras inicia Python.

Si el navegador no se abre, entra manualmente a esa dirección. Para cerrar el servidor, pulsa **Cerrar aplicación** en el menú lateral.

También puedes iniciarlo desde PowerShell, sin usar el archivo `.vbs`:

```powershell
.\.venv\Scripts\python.exe codigo/interfaz.py
```

Si lo iniciaste así, puedes detenerlo con **Ctrl+C**. Si el puerto 8765 está ocupado, usa otro:

```powershell
.\.venv\Scripts\python.exe codigo/interfaz.py --puerto 8766
```

En ese caso abre `http://127.0.0.1:8766`. El doble clic usa siempre el puerto 8765.

## Qué hay en la pantalla

| Sección | Qué muestra |
|---|---|
| **Predicción** | Escribe una frase, elige una RNN o LSTM y consulta cinco palabras posibles. Puedes abrir las etapas del cálculo y descargar los detalles. |
| **Corpus** | Busca oraciones de los tres archivos originales y revisa documento, palabras e IDs. |
| **Preparación** | Muestra cómo se limpiaron y separaron los textos y cómo se creó el vocabulario. Permite preparar una copia sin cambiar los datos originales. |
| **Entrenamiento** | Muestra las curvas y la tabla de las cinco épocas guardadas de cada modelo. |
| **Evaluación** | Compara el bigrama y las diez redes con las mismas oraciones de prueba. |

Para comprobar una predicción, escribe `el presidente del gobierno`. Con la LSTM inicial, la primera sugerencia es `de`. Los porcentajes se reparten entre **todas** las entradas del vocabulario; las cinco palabras visibles no tienen que sumar 100 %.

La aplicación predice **una palabra completa**. No completa letras ni garantiza que una cadena de varias sugerencias forme una oración coherente. La LSTM que aparece al inicio fue elegida por su error de validación **entre las LSTM**; eso no significa que haya superado a las RNN en la prueba final.

## Si algo falla

| Lo que ocurre | Qué revisar |
|---|---|
| `py -3.12` no se reconoce o no encuentra Python | Instala Python 3.12 de 64 bits desde python.org. Cierra y vuelve a abrir PowerShell. |
| El doble clic dice que falta `.venv` | Ejecuta los comandos de instalación desde la carpeta principal, donde está `Abrir Ancora.vbs`. |
| Aparece `No module named torch` o `No module named numpy` | La instalación de bibliotecas no terminó o se usó otro Python. Repite el comando `pip install -r requirements-app.txt ...` con `.\.venv\Scripts\python.exe` como se indica arriba. |
| El navegador no carga | Prueba `http://127.0.0.1:8765`. Revisa `sesiones/interfaz.log` o inicia desde PowerShell para leer el error. |
| El puerto 8765 está ocupado | Cierra otra instancia de Áncora o usa `--puerto 8766` con el comando manual. |
| Faltan modelos, corpus o resultados | Descarga y extrae **todo** el ZIP. Debe haber diez archivos `.pt` en `modelos/` y tres `.conllu` en `datos/originales/`. |
| Las pruebas no terminan en `OK` | Conserva el error completo y comprueba la versión de Python, la instalación de bibliotecas y que el ZIP se extrajo entero. |

## Archivos principales

- `codigo/pln.py`: prepara datos, define y entrena RNN/LSTM, evalúa y predice.
- `codigo/interfaz.py`: servidor local que conecta los datos y modelos con el navegador.
- `codigo/web/`: HTML, CSS y JavaScript de las cinco secciones.
- `codigo/abrir_interfaz.py` y `Abrir Ancora.vbs`: arranque de la interfaz en Windows.
- `datos/originales/`: corpus UD Spanish-AnCora y su licencia.
- `datos/procesados/`: vocabulario y ejemplos numéricos de entrenamiento, validación y prueba.
- `modelos/`: diez modelos entrenados, incluidos en la descarga.
- `resultados/`: historiales y métricas que muestra la aplicación.
- `requirements-app.txt`: dependencias directas para abrir la interfaz.
- `requirements-lock.txt`: entorno completo usado para reproducir también el análisis y las figuras.

La versión de consola sigue disponible con `.\.venv\Scripts\python.exe codigo/demo.py`, pero no hace falta usarla para presentar la aplicación.

El corpus es [UD Spanish-AnCora r2.17](https://github.com/UniversalDependencies/UD_Spanish-AnCora/tree/r2.17). Su documentación y licencia están en `datos/originales/`.
