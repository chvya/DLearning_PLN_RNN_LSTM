"""Aplicación gráfica local: python codigo/interfaz.py (sin dependencias nuevas)."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import unicodedata
from urllib.parse import urlparse, parse_qs
import webbrowser

import torch
import torch.nn.functional as F
import pln

ROOT = pln.ROOT
WEB = ROOT / 'codigo/web'
LOCK = threading.Lock()
JOBS = {}
SESSION_TOKEN = secrets.token_urlsafe(32)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def models():
    result = []
    for path in sorted(pln.MODELS.glob('*.pt')):
        info = pln.OUT / f'{path.stem}_entrenamiento.json'
        if info.exists():
            row = read_json(info)
            result.append({'id': path.stem, **row})
    return result


def overview():
    available = models()
    candidates = [r for r in available if r['config']['kind'] == 'lstm' and not r['config']['piloto']]
    latest_jobs = {}
    for saved in sorted((ROOT / 'sesiones').glob('*/ejecucion.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            job = read_json(saved)
            latest_jobs.setdefault(job['accion'], job)
        except (OSError, ValueError, KeyError):
            continue
        if len(latest_jobs) == 2: break
    return {'modelos': available,
            'ejecuciones': latest_jobs,
            'recomendado': min(candidates, key=lambda r: r['mejor_val_nll'])['id'] if candidates else None,
            'resumen': read_json(pln.DATA / 'resumen.json'),
            'evaluacion': read_json(pln.OUT / 'evaluacion_test.json'),
            'comparacion': read_json(pln.OUT / 'resumen_comparacion.json')}


@lru_cache(maxsize=12)
def get_model(name):
    if name not in {r['id'] for r in models()}:
        raise ValueError('Modelo no disponible. Seleccione uno de la lista.')
    return pln.load_model(pln.MODELS / f'{name}.pt')


@torch.inference_mode()
def trace(text, name):
    if not isinstance(text, str) or len(text) > 5000:
        raise ValueError('El texto debe tener como máximo 5000 caracteres.')
    start = time.perf_counter()
    model, checkpoint = get_model(name)
    cfg, vocab = checkpoint['config'], checkpoint['vocab']
    index = {word: i for i, word in enumerate(vocab)}
    words = pln.tokens(text)
    all_ids = [pln.BOS] + [index.get(w, pln.UNK) for w in words]
    ids = all_ids[-cfg['context']:]
    retained = (['<bos>'] + words)[-cfg['context']:]
    x = torch.tensor([ids])
    emb = model.embedding(x)
    h = torch.zeros(1, cfg['hidden'])
    c = torch.zeros_like(h)
    states = []
    r = model.recurrent
    for t, ident in enumerate(ids):
        gates = F.linear(emb[:, t], r.weight_ih_l0, r.bias_ih_l0) + F.linear(h, r.weight_hh_l0, r.bias_hh_l0)
        extra = {}
        if cfg['kind'] == 'lstm':
            i, f, g, o = gates.chunk(4, dim=1)
            i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
            c = f * c + i * g
            h = o * c.tanh()
            extra = {'entrada': i[0].tolist(), 'olvido': f[0].tolist(),
                     'candidato': g[0].tolist(), 'salida': o[0].tolist(), 'celda': c[0].tolist()}
        else:
            h = gates.tanh()
        states.append({'token': retained[t], 'id': ident, 'embedding': emb[0, t].tolist(),
                       'oculto': h[0].tolist(), **extra})
    logits = model(x, torch.tensor([len(ids)]))[0]
    manual = model.output(h)[0]
    if not torch.allclose(manual[3:], logits[3:], rtol=1e-4, atol=1e-5):
        raise RuntimeError('La traza no coincide con la salida del modelo.')
    probs = logits.softmax(-1)
    order = torch.argsort(probs, descending=True).tolist()
    distribution = [{'palabra': vocab[i], 'id': i, 'logit': float(logits[i]),
                     'probabilidad': float(probs[i])} for i in order if i >= 3]
    return {'modelo': name, 'config': cfg, 'epoca': checkpoint['epoch'], 'entrada': text,
            'normalizado': unicodedata.normalize('NFC', text).lower(), 'tokens': words,
            'ids': [index.get(w, pln.UNK) for w in words], 'contexto_ids': ids,
            'descartados': max(0, len(all_ids) - len(ids)), 'estados': states,
            'desconocidas': [w for w in words if w not in index],
            'masa_unk': float(probs[pln.UNK]), 'sugerencias': distribution[:5],
            'distribucion': distribution, 'parametros': sum(p.numel() for p in model.parameters()),
            'milisegundos': (time.perf_counter() - start) * 1000}


@lru_cache(maxsize=1)
def corpus():
    rows, seen, counts = [], set(), Counter()
    index = {w: i for i, w in enumerate(read_json(pln.DATA / 'vocabulario.json'))}
    files = []
    # Mismo orden que corpus_records(): determina qué duplicado se conserva.
    for part in ('dev', 'test', 'train'):
        path = ROOT / 'datos/originales' / f'es_ancora-ud-{part}.conllu'
        doc = sent = None
        total = 0
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.startswith('# newdoc id = '): doc = line.split(' = ', 1)[1]
            elif line.startswith('# sent_id = '): sent = line.split(' = ', 1)[1]
            elif line.startswith('# text = '):
                text = line.split(' = ', 1)[1]
                words = pln.tokens(text)
                if not words: continue
                signature = ' '.join(words)
                duplicate = signature in seen
                seen.add(signature)
                split = pln.split_document(doc)
                rows.append({'texto': text, 'tokens': words, 'ids': [index.get(w, 1) for w in words],
                             'documento': doc, 'oracion': sent, 'particion': split,
                             'duplicada': duplicate, 'archivo': path.name})
                if split == 'train' and not duplicate: counts.update(words)
                total += 1
        files.append({'nombre': path.name, 'bytes': path.stat().st_size, 'oraciones': total})
    return rows, files, counts


def corpus_page(query='', split='', page=0):
    rows, files, counts = corpus()
    filtered = [r for r in rows if (not split or r['particion'] == split) and query.lower() in r['texto'].lower()]
    page = max(0, page)
    return {'archivos': files, 'total': len(filtered), 'pagina': page,
            'filas': filtered[page * 12:(page + 1) * 12],
            'frecuencias': counts.most_common(24), 'cargadas': len(rows)}


def vocabulary(query=''):
    _, _, counts = corpus()
    vocab = read_json(pln.DATA / 'vocabulario.json')
    rows = [{'palabra': w, 'id': i, 'frecuencia': counts[w]} for i, w in enumerate(vocab) if query.lower() in w]
    return {'total': len(rows), 'filas': rows[:100]}


def job_snapshot(ident):
    with LOCK:
        if ident not in JOBS:
            if isinstance(ident, str) and len(ident) == 12 and all(c in '0123456789abcdef' for c in ident):
                saved = ROOT / 'sesiones' / ident / 'ejecucion.json'
                if saved.is_file(): return read_json(saved)
            raise ValueError('La ejecución no existe o el servidor se reinició antes de finalizarla.')
        return json.loads(json.dumps(JOBS[ident]))


def launch_job(options):
    action = options.get('accion')
    if action not in ('preparar', 'entrenar'): raise ValueError('Operación no válida.')
    kind = options.get('modelo', 'lstm')
    context = int(options.get('contexto', 12))
    epochs = int(options.get('epocas', 1))
    if kind not in ('rnn', 'lstm') or context not in (4, 12, 24) or not 1 <= epochs <= 5:
        raise ValueError('Configuración de entrenamiento no válida.')
    with LOCK:
        if any(j['estado'] == 'ejecutando' for j in JOBS.values()):
            raise ValueError('Ya hay una ejecución en curso. Espere a que termine.')
        ident = secrets.token_hex(6)
        JOBS[ident] = {'id': ident, 'estado': 'ejecutando', 'accion': action, 'registro': [],
                       'eventos': [], 'historial': [], 'resultado': None}

    def worker():
        dest = ROOT / 'sesiones' / ident
        def log(message):
            with LOCK: JOBS[ident]['registro'].append(message)
        try:
            (dest / 'codigo').mkdir(parents=True)
            shutil.copy2(ROOT / 'codigo/pln.py', dest / 'codigo/pln.py')
            if action == 'preparar':
                log('Leyendo los tres archivos CoNLL-U del corpus local…')
                shutil.copytree(ROOT / 'datos/originales', dest / 'datos/originales')
                log('Normalización → deduplicación → partición por documento → vocabulario → ventanas.')
                command = ['preparar']
            else:
                log('Cargando una copia de los datos procesados. Los resultados originales se conservan.')
                shutil.copytree(pln.DATA, dest / 'datos/procesados')
                command = ['entrenar', '--modelo', kind, '--contexto', str(context), '--epocas', str(epochs), '--limite', '2048']
                log(f'Piloto real: {kind.upper()}, contexto {context}, {epochs} época(s), 2048 objetivos por partición, CPU.')
            with subprocess.Popen([sys.executable, '-u', '-X', 'utf8', str(dest / 'codigo/pln.py'), *command, '--progreso'],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  encoding='utf-8', errors='replace', cwd=dest) as process:
                for line in process.stdout:
                    if line.startswith('PROGRESO '):
                        event = json.loads(line[len('PROGRESO '):])
                        with LOCK:
                            JOBS[ident]['eventos'].append(event)
                            if event['etapa'] == 'epoca_completada': JOBS[ident]['historial'].append(event)
                        log(' · '.join(f'{k}: {v}' for k, v in event.items() if k != 'val'))
                    else:
                        log(line.rstrip())
                if process.wait() != 0: raise RuntimeError('El proceso terminó con un error. Revise el registro.')
            if action == 'preparar':
                result = read_json(dest / 'datos/procesados/resumen.json')
            else:
                result = read_json(next((dest / 'resultados').glob('*_historial.json')))
            log(f'Finalizado. Archivos guardados en sesiones/{ident}.')
            with LOCK: JOBS[ident].update(estado='completado', resultado=result)
        except Exception as exc:
            log(str(exc))
            with LOCK: JOBS[ident]['estado'] = 'error'
        finally:
            # El registro queda disponible en disco aunque se cierre la aplicación.
            if dest.exists():
                pln.save_json(dest / 'ejecucion.json', job_snapshot(ident))
    threading.Thread(target=worker, daemon=True).start()
    return {'id': ident}


class LocalServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def respond(self, status, value, mime='application/json; charset=utf-8'):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8') if mime.startswith('application/json') else value
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path == '/api/inicio': self.respond(200, {**overview(), 'token': SESSION_TOKEN})
            elif url.path == '/api/corpus':
                self.respond(200, corpus_page(q.get('q', [''])[0], q.get('particion', [''])[0], int(q.get('pagina', ['0'])[0])))
            elif url.path == '/api/vocabulario': self.respond(200, vocabulary(q.get('q', [''])[0]))
            elif url.path == '/api/historial':
                name = q.get('modelo', [''])[0]
                if name not in {r['id'] for r in models()}: raise ValueError('Modelo no válido.')
                self.respond(200, read_json(pln.OUT / f'{name}_historial.json'))
            elif url.path == '/api/trabajo': self.respond(200, job_snapshot(q.get('id', [''])[0]))
            else:
                files = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8'), '/style.css': ('style.css', 'text/css; charset=utf-8')}
                if url.path not in files:
                    self.respond(404, {'error': 'Recurso no encontrado.'}); return
                file, mime = files[url.path]
                self.respond(200, (WEB / file).read_bytes(), mime)
        except (ValueError, KeyError) as exc: self.respond(400, {'error': str(exc)})
        except Exception as exc: self.respond(500, {'error': f'No se pudo completar la operación: {exc}'})

    def do_POST(self):
        try:
            if self.headers.get('X-Session-Token') != SESSION_TOKEN:
                self.respond(403, {'error': 'Sesión no válida. Recargue la aplicación.'}); return
            length = int(self.headers.get('Content-Length', 0))
            if length < 1 or length > 100000: raise ValueError('Solicitud vacía o demasiado grande.')
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict): raise ValueError('Solicitud no válida.')
            if self.path == '/api/predecir':
                with LOCK: result = trace(body.get('texto', ''), body.get('modelo', ''))
                self.respond(200, result)
            elif self.path == '/api/ejecutar': self.respond(202, launch_job(body))
            elif self.path == '/api/cerrar':
                with LOCK:
                    if any(j['estado'] == 'ejecutando' for j in JOBS.values()):
                        raise ValueError('Espere a que finalice la ejecución antes de cerrar el laboratorio.')
                self.respond(200, {'mensaje': 'Aplicación cerrada.'})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else: self.respond(404, {'error': 'Operación no encontrada.'})
        except (ValueError, KeyError, TypeError) as exc: self.respond(400, {'error': str(exc)})
        except Exception as exc: self.respond(500, {'error': f'No se pudo completar la operación: {exc}'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--puerto', type=int, default=8765)
    parser.add_argument('--sin-navegador', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        server = LocalServer(('127.0.0.1', args.puerto), Handler)
    except OSError as exc:
        print(f'No se pudo abrir el puerto {args.puerto}: {exc}. Use --puerto con otro número.')
        return 1
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Aplicación disponible en {url}', flush=True)
    if not args.sin_navegador: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
    return 0


if __name__ == '__main__': sys.exit(main())
