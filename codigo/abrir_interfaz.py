"""Lanzador gráfico: reutiliza una instancia abierta y muestra errores en ventana."""
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen
import webbrowser


def main():
    root = Path(__file__).resolve().parents[1]
    url = 'http://127.0.0.1:8765'

    def running():
        try:
            with urlopen(url + '/api/inicio', timeout=1) as response:
                data = json.load(response)
            return 'modelos' in data and 'recomendado' in data
        except Exception:
            return False

    if running():
        webbrowser.open(url)
        return
    logs = root / 'sesiones'
    logs.mkdir(exist_ok=True)
    with (logs / 'interfaz.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-X', 'utf8', str(root / 'codigo/interfaz.py'), '--sin-navegador'],
                                   stdout=log, stderr=log, cwd=root)
    for _ in range(60):
        if running():
            webbrowser.open(url)
            return
        if process.poll() is not None: break
        time.sleep(.5)
    from tkinter import Tk, messagebox
    window = Tk()
    window.withdraw()
    messagebox.showerror('Áncora', 'No se pudo iniciar la aplicación. Revisa sesiones/interfaz.log. '
                         'El puerto 8765 puede estar ocupado o pueden faltar dependencias.')
    window.destroy()


if __name__ == '__main__': main()
