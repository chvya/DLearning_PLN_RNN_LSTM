"""Verificación de la interfaz contra el motor original y su evidencia."""
import json
from pathlib import Path
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import torch
import interfaz as ui
import pln


HAS_ARTIFACTS = (pln.OUT / 'evaluacion_test.json').exists() and len(list(pln.MODELS.glob('*.pt'))) >= 10


@unittest.skipUnless(HAS_ARTIFACTS, 'La integración gráfica requiere los modelos y la evaluación terminados.')
class TraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_trace_matches_original_for_every_model(self):
        for row in ui.models():
            with self.subTest(model=row['id']):
                text = 'el presidente del gobierno'
                actual = ui.trace(text, row['id'])
                expected = pln.predict(text, pln.MODELS / (row['id'] + '.pt'))
                self.assertEqual([r['palabra'] for r in actual['sugerencias']],
                                 [r['palabra'] for r in expected['sugerencias']])
                for a, b in zip(actual['sugerencias'], expected['sugerencias']):
                    self.assertAlmostEqual(a['probabilidad'], b['probabilidad'], places=7)
                self.assertAlmostEqual(sum(r['probabilidad'] for r in actual['distribucion']) + actual['masa_unk'], 1, places=6)

    def test_empty_unknown_numeric_and_truncated_inputs(self):
        for kind in ('rnn', 'lstm'):
            name = f'{kind}_c4_s17'
            for text in ('', '!!!', 'ESPOCH xyzabc 2026', 'el presidente del gobierno de la ciudad de madrid'):
                with self.subTest(kind=kind, text=text):
                    result = ui.trace(text, name)
                    self.assertLessEqual(len(result['estados']), 4)
                    self.assertEqual(result['tokens'], pln.tokens(text))
                    self.assertEqual(len(result['estados'][0]['embedding']), 48)
                    self.assertEqual(len(result['estados'][0]['oculto']), 64)
                    if kind == 'lstm':
                        for s in result['estados']:
                            self.assertTrue(all(0 <= v <= 1 for v in s['olvido']))
                    if not pln.tokens(text): self.assertEqual(result['contexto_ids'], [pln.BOS])
        with self.assertRaises(ValueError): ui.trace('x' * 5001, 'lstm_c4_s17')
        with self.assertRaises(ValueError): ui.trace('hola', '../../secreto')

    def test_corpus_matches_prepared_partitions_and_vocabulary(self):
        rows, files, counts = ui.corpus()
        summary = ui.overview()['resumen']
        self.assertEqual(len(files), 3)
        self.assertEqual(sum(r['duplicada'] for r in rows), summary['duplicados_excluidos'])
        for split, info in summary['particiones'].items():
            subset = [r for r in rows if r['particion'] == split and not r['duplicada']]
            self.assertEqual(len(subset), info['oraciones'])
            self.assertEqual(sum(len(r['tokens']) for r in subset), info['palabras_totales'])
        vocab = pln.SPECIAL + sorted(counts, key=lambda w: (-counts[w], w))[:2997]
        self.assertEqual(vocab, ui.read_json(pln.DATA / 'vocabulario.json'))
        self.assertEqual(ui.corpus_page('not-a-spanish-sentence-xyz')['total'], 0)

    def test_recommendation_is_selected_by_validation(self):
        o = ui.overview()
        candidates = [m for m in o['modelos'] if m['config']['kind'] == 'lstm' and not m['config']['piloto']]
        self.assertEqual(o['recomendado'], min(candidates, key=lambda m: m['mejor_val_nll'])['id'])

    def test_saved_job_path_is_restricted(self):
        for ident in ('../proyecto', '../../README.md', 'no-es-una-sesion'):
            with self.subTest(ident=ident), self.assertRaises(ValueError):
                ui.job_snapshot(ident)


@unittest.skipUnless(HAS_ARTIFACTS, 'La integración gráfica requiere los modelos y la evaluación terminados.')
class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ui.ThreadingHTTPServer(('127.0.0.1', 0), ui.Handler)
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, body=None, token=ui.SESSION_TOKEN):
        request = Request(self.url + path, data=json.dumps(body).encode() if body is not None else None,
                          headers={'X-Session-Token': token, 'Content-Type': 'application/json'})
        return urlopen(request, timeout=30)

    def test_application_and_assets(self):
        for path in ('/', '/app.js', '/style.css', '/api/inicio', '/api/corpus?pagina=0',
                     '/api/vocabulario?q=gobierno', '/api/historial?modelo=lstm_c12_s17'):
            with self.subTest(path=path), self.request(path) as response:
                self.assertEqual(response.status, 200)
                self.assertGreater(len(response.read()), 0)

    def test_prediction_and_bad_requests(self):
        with self.request('/api/predecir', {'texto': 'la universidad de', 'modelo': 'lstm_c12_s17'}) as response:
            self.assertEqual(len(json.load(response)['sugerencias']), 5)
        for path, body, token, code in [
            ('/api/predecir', {'texto': 'hola', 'modelo': 'lstm_c12_s17'}, 'incorrecto', 403),
            ('/api/predecir', {'texto': 17, 'modelo': 'lstm_c12_s17'}, ui.SESSION_TOKEN, 400),
            ('/api/predecir', {'texto': 'hola', 'modelo': '../x'}, ui.SESSION_TOKEN, 400),
            ('/api/ejecutar', {'accion': 'entrenar', 'epocas': 9}, ui.SESSION_TOKEN, 400),
            ('/../../proyecto.json', None, ui.SESSION_TOKEN, 404),
        ]:
            with self.subTest(path=path, body=body), self.assertRaises(HTTPError) as caught:
                self.request(path, body, token)
            self.assertEqual(caught.exception.code, code)


if __name__ == '__main__': unittest.main()
