"""Demo interactiva de siguiente palabra. Ejecutar: python codigo/demo.py.

No necesita conexión una vez descargados los modelos. Escribir /salir termina.
Las probabilidades mostradas conservan la masa de los símbolos ocultos.
"""
import sys
from pln import predict


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print('Grupo 5 | Autocompletado LSTM en español')
    print('Escriba palabras completas. /salir termina. No completa letras parciales.')
    while True:
        try:
            text = input('\nContexto> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nFin de la demostración.')
            break
        if text == '/salir':
            break
        result = predict(text)
        print('Modelo seleccionado por validación:', result['modelo'])
        print('Palabras desconocidas:', ', '.join(result['desconocidas']) or 'ninguna')
        print(f"Probabilidad de UNK: {100 * result['masa_unk']:.2f}%")
        for i, row in enumerate(result['sugerencias'], 1):
            print(f"  {i}. {row['palabra']:<20} {100 * row['probabilidad']:6.2f}%")


if __name__ == '__main__':
    main()
