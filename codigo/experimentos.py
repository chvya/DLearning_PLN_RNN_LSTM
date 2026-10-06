"""Ejecuta el protocolo fijado sin consultar el conjunto de prueba."""
from pln import train

if __name__ == '__main__':
    for seed in (17,29,43):
        for kind in ('rnn','lstm'):
            train(kind, seed=seed, context=12, epochs=5)
    # Sensibilidad de contexto: exploratoria, una semilla, sin inferencia estadística.
    for context in (4,24):
        for kind in ('rnn','lstm'):
            train(kind, seed=17, context=context, epochs=5)
