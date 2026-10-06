"""Autocompletado por palabras en español con RNN y LSTM.

Preparar: python codigo/pln.py preparar
Entrenar: python codigo/pln.py entrenar --modelo lstm --semilla 17
Evaluar:  python codigo/pln.py evaluar
Demo:     python codigo/pln.py predecir --texto "el presidente del gobierno"

"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import time
import unicodedata

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datos/procesados'
OUT = ROOT / 'resultados'
MODELS = ROOT / 'modelos'
PAD, UNK, BOS = 0, 1, 2
SPECIAL = ['<pad>', '<unk>', '<bos>']
WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*|\d+(?:[.,]\d+)*", re.UNICODE)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def tokens(text):
    """Minúsculas NFC, tildes conservadas y números normalizados.

    Se omite puntuación: la tarea predice palabras, no signos ni caracteres.
    No se eliminan stopwords ni se aplica lematización.
    """
    items = WORD.findall(unicodedata.normalize('NFC', text).lower())
    return ['<num>' if x[0].isdigit() else x for x in items]


def corpus_records():
    """Lee texto original, no FORM: evita desdoblar 'del' o 'al'."""
    for partition in ('train', 'dev', 'test'):
        required = ROOT / 'datos/originales' / f'es_ancora-ud-{partition}.conllu'
        if not required.is_file():
            raise FileNotFoundError(f'Corpus incompleto: falta {required.name}')
    for path in sorted((ROOT / 'datos/originales').glob('*.conllu')):
        doc = None
        sent = None
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.startswith('# newdoc id = '):
                doc = line.split(' = ', 1)[1]
            elif line.startswith('# sent_id = '):
                sent = line.split(' = ', 1)[1]
            elif line.startswith('# text = '):
                if not doc or not sent:
                    raise ValueError('Faltan identificadores de documento/oración')
                words = tokens(line.split(' = ', 1)[1])
                if words:
                    yield {'documento': doc, 'oracion': sent, 'palabras': words}


def split_document(doc):
    # Asignación invariable por documento, independiente de los modelos.
    number = int(hashlib.sha256(('grupo5-v1:' + doc).encode()).hexdigest()[:8], 16) % 100
    return 'train' if number < 80 else ('val' if number < 90 else 'test')


def prepare(vocab_size=3000, max_train=100000):
    records = {'train': [], 'val': [], 'test': []}
    seen = set()
    removed = 0
    # Deduplicación global exacta después de normalizar; no comparación semántica.
    for row in corpus_records():
        signature = ' '.join(row['palabras'])
        if signature in seen:
            removed += 1
            continue
        seen.add(signature)
        records[split_document(row['documento'])].append(row)
    if not all(records.values()):
        raise ValueError('No hay datos suficientes; descargue primero el corpus')
    counts = Counter(w for row in records['train'] for w in row['palabras'])
    ranked = sorted(counts, key=lambda w: (-counts[w], w))
    vocab = SPECIAL + ranked[:vocab_size - len(SPECIAL)]
    index = {w: i for i, w in enumerate(vocab)}
    save_json(DATA / 'vocabulario.json', vocab)
    save_json(DATA / 'oraciones.json', records)
    summary = {'vocabulario': len(vocab), 'duplicados_excluidos': removed,
               'regla_particion': 'SHA256 grupo5-v1:documento, intervalos 80/10/10',
               'contexto_maximo': 24, 'max_train': max_train, 'particiones': {}}
    docsets = {}
    for split, rows in records.items():
        docsets[split] = {r['documento'] for r in rows}
        examples, targets, lengths, sent_ids, positions = [], [], [], [], []
        for rid, row in enumerate(rows):
            ids = [index.get(w, UNK) for w in row['palabras']]
            for t, target in enumerate(ids):
                history = ([BOS] + ids[:t])[-24:]
                examples.append(history + [PAD] * (24 - len(history)))
                targets.append(target)
                lengths.append(len(history))
                sent_ids.append(rid)
                positions.append(t)
        x, y, lens = np.asarray(examples, np.int64), np.asarray(targets, np.int64), np.asarray(lengths, np.int64)
        chosen = np.arange(len(y))
        if split == 'train' and len(y) > max_train:
            chosen = np.sort(np.random.default_rng(20260929).choice(len(y), max_train, replace=False))
        np.savez_compressed(DATA / f'{split}.npz', x=x[chosen], y=y[chosen], lengths=lens[chosen],
                            sentence=np.asarray(sent_ids)[chosen], position=np.asarray(positions)[chosen])
        summary['particiones'][split] = {'documentos': len(docsets[split]), 'oraciones': len(rows),
            'palabras_totales': len(y), 'objetivos_usados': len(chosen),
            'oov_objetivos': int((y[chosen] == UNK).sum()), 'tasa_oov': float((y[chosen] == UNK).mean())}
    assert not (docsets['train'] & docsets['val'] or docsets['train'] & docsets['test'] or docsets['val'] & docsets['test'])
    save_json(DATA / 'documentos.json', {k: sorted(v) for k, v in docsets.items()})
    summary['sin_documentos_compartidos'] = True
    save_json(DATA / 'resumen.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_split(split, context):
    arrays = np.load(DATA / f'{split}.npz')
    source, lens = arrays['x'], arrays['lengths']
    x = np.zeros((len(source), context), dtype=np.int64)
    for i, length in enumerate(lens):
        sequence = source[i, max(0, int(length) - context):int(length)]
        x[i, :len(sequence)] = sequence
    return torch.from_numpy(x), torch.from_numpy(arrays['y'].copy()), torch.from_numpy(np.minimum(lens, context).copy())


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, kind='lstm', embedding=48, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding, padding_idx=PAD)
        recurrent = nn.LSTM if kind == 'lstm' else nn.RNN
        self.recurrent = recurrent(embedding, hidden, batch_first=True)
        self.dropout = nn.Dropout(0.15)
        self.output = nn.Linear(hidden, vocab_size)

    def forward(self, x, lengths):
        values, _ = self.recurrent(self.embedding(x))
        # Solo la última posición real; el relleno posterior no afecta su estado.
        last = values[torch.arange(x.shape[0], device=x.device), lengths - 1]
        logits = self.output(self.dropout(last))
        # PAD y BOS nunca son palabras objetivo.
        logits[:, PAD] = -1e9
        logits[:, BOS] = -1e9
        return logits


def metrics_from_logits(logits, y):
    loss = nn.functional.cross_entropy(logits, y, reduction='sum').item()
    top = logits.topk(5, dim=1).indices
    # UNK nunca cuenta como acierto léxico: no identifica la palabra original.
    known = y != UNK
    correct1 = ((top[:, 0] == y) & known).sum().item()
    correct5 = ((top == y[:, None]).any(dim=1) & known).sum().item()
    return loss, correct1, correct5


@torch.inference_mode()
def evaluate_model(model, data, batch_size=512):
    model.eval()
    x, y, lengths = data
    total = np.zeros(3)
    for start in range(0, len(y), batch_size):
        total += metrics_from_logits(model(x[start:start+batch_size], lengths[start:start+batch_size]), y[start:start+batch_size])
    n = len(y)
    nll = float(total[0] / n)
    return {'n': n, 'nll': nll, 'perplejidad': math.exp(nll),
            'top1': float(total[1] / n), 'top5': float(total[2] / n)}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def train(kind, seed=17, context=12, epochs=5, limit=None):
    seed_everything(seed)
    name = f'{kind}_c{context}_s{seed}' + ('_piloto' if limit else '')
    path = MODELS / f'{name}.pt'
    if path.exists():
        raise FileExistsError(f'Ya existe {path}; use otra semilla o preserve el resultado antes de repetir')
    vocab = json.loads((DATA / 'vocabulario.json').read_text(encoding='utf-8'))
    train_data = load_split('train', context)
    val_data = load_split('val', context)
    if limit:
        train_data = tuple(a[:limit] for a in train_data)
        val_data = tuple(a[:limit] for a in val_data)
    model = LanguageModel(len(vocab), kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    config = dict(kind=kind, seed=seed, context=context, epochs=epochs, embedding=48, hidden=64,
                  vocab_size=len(vocab), batch_size=512, learning_rate=0.003, dropout=0.15,
                  clip_norm=1.0, train_n=len(train_data[1]), torch=str(torch.__version__), numpy=np.__version__,
                  python=platform.python_version(), platform=platform.platform(), processor=platform.processor(),
                  threads=2, piloto=bool(limit))
    MODELS.mkdir(exist_ok=True)
    history, best = [], float('inf')
    begin = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        order = torch.randperm(len(train_data[1]))
        total_loss = 0.0
        tick = time.perf_counter()
        for ids in order.split(512):
            x, y, lengths = [a[ids] for a in train_data]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, lengths)
            loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Pérdida no finita')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
        validation = evaluate_model(model, val_data)
        row = {'epoca': epoch, 'train_nll': total_loss / len(order), 'val': validation,
               'segundos': time.perf_counter() - tick}
        history.append(row)
        if validation['nll'] < best:
            best = validation['nll']
            torch.save({'state_dict': model.state_dict(), 'config': config, 'epoch': epoch, 'vocab': vocab}, path)
        save_json(OUT / f'{name}_historial.json', history)
        print(name, json.dumps(row), flush=True)
    summary = {'config': config, 'parametros': sum(p.numel() for p in model.parameters()),
               'segundos_entrenamiento': time.perf_counter() - begin, 'mejor_val_nll': best}
    save_json(OUT / f'{name}_entrenamiento.json', summary)


def load_model(path):
    # Compatibilidad con los checkpoints locales iniciales: TorchVersion es
    # una subclase de str. Se autoriza solo ese tipo, manteniendo weights_only.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    model = LanguageModel(cfg['vocab_size'], cfg['kind'], cfg['embedding'], cfg['hidden'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


def fit_bigram():
    x, y, lengths = load_split('train', 1)
    v = len(json.loads((DATA / 'vocabulario.json').read_text(encoding='utf-8')))
    counts = np.zeros((v, v), dtype=np.float64)
    np.add.at(counts, (x[:, 0].numpy(), y.numpy()), 1)
    # Probabilidad base suavizada; PAD y BOS excluidos de la distribución.
    unigram = counts.sum(axis=0) + 0.1
    unigram[[PAD, BOS]] = 0
    unigram /= unigram.sum()
    return counts, unigram


def bigram_probs(counts, unigram, previous, alpha=10.0):
    selected = counts[previous]
    return (selected + alpha * unigram) / (selected.sum(axis=1, keepdims=True) + alpha)


def evaluate_baseline(counts, unigram, split, alpha):
    x, y, _ = load_split(split, 1)
    sums = np.zeros(3)
    for begin in range(0, len(y), 256):
        target = y[begin:begin+256]
        probs = bigram_probs(counts, unigram, x[begin:begin+256, 0].numpy(), alpha)
        sums += metrics_from_logits(torch.from_numpy(np.log(np.maximum(probs, 1e-300))), target)
    nll = float(sums[0] / len(y))
    return dict(n=len(y), nll=nll, perplejidad=math.exp(nll), top1=float(sums[1]/len(y)), top5=float(sums[2]/len(y)))


def evaluate_all():
    torch.set_num_threads(2)
    destination = OUT / 'evaluacion_test.json'
    if destination.exists():
        raise FileExistsError('La prueba ya fue evaluada; conserve los resultados y no ajuste modelos mirando prueba')
    results = []
    counts, unigram = fit_bigram()
    search = {a: evaluate_baseline(counts, unigram, 'val', a) for a in (1.0, 10.0, 100.0)}
    alpha = min(search, key=lambda a: search[a]['nll'])
    results.append({'nombre': 'bigrama', 'modelo': 'bigrama', 'alpha': alpha,
                    'validacion': search[alpha], 'test': evaluate_baseline(counts, unigram, 'test', alpha)})
    save_json(OUT / 'bigrama_validacion.json', search)
    for path in sorted(MODELS.glob('*.pt')):
        if 'piloto' in path.stem:
            continue
        model, ckpt = load_model(path)
        cfg = ckpt['config']
        data = load_split('test', cfg['context'])
        test = evaluate_model(model, data)
        # Medición de latencia: 10 calentamientos, 100 consultas, lote 1.
        durations = []
        with torch.inference_mode():
            for j in range(110):
                tick = time.perf_counter()
                model(data[0][j:j+1], data[2][j:j+1]).softmax(-1).topk(5)
                if j >= 10:
                    durations.append(1000*(time.perf_counter()-tick))
        training = json.loads((OUT / f'{path.stem}_entrenamiento.json').read_text(encoding='utf-8'))
        results.append({'nombre': path.stem, 'modelo': cfg['kind'], 'semilla': cfg['seed'],
                        'contexto': cfg['context'], 'mejor_epoca': ckpt['epoch'], 'test': test,
                        'latencia_mediana_ms': float(np.median(durations)),
                        'latencia_p95_ms': float(np.percentile(durations, 95)), **training})
        print(path.stem, test, flush=True)
    save_json(destination, results)


@torch.inference_mode()
def predict(text, path=None):
    torch.set_num_threads(2)
    if path is None:
        # La demo se escoge por validación, nunca por prueba.
        candidates = []
        for file in OUT.glob('lstm*_entrenamiento.json'):
            info = json.loads(file.read_text(encoding='utf-8'))
            if not info['config']['piloto']:
                candidates.append((info['mejor_val_nll'], MODELS / file.name.replace('_entrenamiento.json', '.pt')))
        if not candidates:
            raise FileNotFoundError('No hay LSTM entrenada')
        path = min(candidates)[1]
    model, ckpt = load_model(path)
    vocab = ckpt['vocab']
    index = {w:i for i,w in enumerate(vocab)}
    words = tokens(text)
    ids = ([BOS] + [index.get(w, UNK) for w in words])[-ckpt['config']['context']:]
    x = torch.tensor([ids])
    probability = model(x, torch.tensor([len(ids)])).softmax(-1)[0]
    # Conservar probabilidades originales: no renormalizar al ocultar símbolos.
    order = torch.argsort(probability, descending=True)
    suggestions = [{'palabra': vocab[i], 'probabilidad': float(probability[i])} for i in order.tolist() if i >= 3][:5]
    return {'modelo': Path(path).name, 'entrada': text, 'tokens': words,
            'desconocidas': [w for w in words if w not in index],
            'masa_unk': float(probability[UNK]), 'sugerencias': suggestions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('accion', choices=['preparar','entrenar','evaluar','predecir'])
    parser.add_argument('--modelo', choices=['rnn','lstm'], default='lstm')
    parser.add_argument('--semilla', type=int, default=17)
    parser.add_argument('--contexto', type=int, choices=[4,12,24], default=12)
    parser.add_argument('--epocas', type=int, default=5)
    parser.add_argument('--limite', type=int)
    parser.add_argument('--texto', default='el presidente del gobierno')
    args = parser.parse_args()
    if args.accion == 'preparar': prepare()
    elif args.accion == 'entrenar': train(args.modelo,args.semilla,args.contexto,args.epocas,args.limite)
    elif args.accion == 'evaluar': evaluate_all()
    else: print(json.dumps(predict(args.texto), ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
