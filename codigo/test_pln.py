"""Pruebas de integridad del pipeline, sin entrenar el modelo final."""
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from pln import (DATA, BOS, PAD, UNK, LanguageModel, tokens, load_split,
                 metrics_from_logits, bigram_probs, load_model)

class PipelineTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(tokens('¡El niño llegó en 2026!'), ['el','niño','llegó','en','<num>'])
        self.assertEqual(tokens(''), [])
        self.assertEqual(tokens('del gobierno'), ['del','gobierno'])

    def test_document_disjointness(self):
        parts = json.loads((DATA/'documentos.json').read_text(encoding='utf-8'))
        for a,b in [('train','val'),('train','test'),('val','test')]:
            self.assertFalse(set(parts[a]) & set(parts[b]))

    def test_targets_match_original_and_no_future(self):
        rows = json.loads((DATA/'oraciones.json').read_text(encoding='utf-8'))['val']
        vocab = json.loads((DATA/'vocabulario.json').read_text(encoding='utf-8'))
        ids = {word:i for i,word in enumerate(vocab)}
        original = np.load(DATA/'val.npz')
        x,y,lengths = load_split('val',12)
        for i in np.linspace(0,len(y)-1,100,dtype=int):
            row = rows[int(original['sentence'][i])]['palabras']
            pos = int(original['position'][i])
            expected = ([BOS]+[ids.get(w,UNK) for w in row[:pos]])[-12:]
            self.assertEqual(x[i,:lengths[i]].tolist(),expected)
            self.assertEqual(y[i].item(),ids.get(row[pos],UNK))
            self.assertTrue((x[i,lengths[i]:] == PAD).all())

    def test_padding_does_not_change_prediction(self):
        torch.manual_seed(7)
        for kind in ('rnn','lstm'):
            model = LanguageModel(8,kind).eval()
            a = model(torch.tensor([[2,3,4]]),torch.tensor([3]))
            b = model(torch.tensor([[2,3,4,0,0]]),torch.tensor([3]))
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)

    def test_unknown_is_not_lexical_hit(self):
        logits = torch.tensor([[0.,10.,0.,0.,0.,0.]])
        loss,one,five = metrics_from_logits(logits,torch.tensor([UNK]))
        self.assertEqual((one,five),(0,0))
        self.assertTrue(math.isfinite(loss))

    def test_bigram_distribution(self):
        counts = np.zeros((8,8)); counts[3,4]=5
        base = np.array([0,.2,0,.2,.2,.1,.1,.2])
        p = bigram_probs(counts,base,np.array([3,7]),10)
        np.testing.assert_allclose(p.sum(1),1)
        self.assertTrue(np.all(p[:,[0,2]]==0))
        np.testing.assert_allclose(p[1],base)

    def test_save_reload(self):
        model = LanguageModel(8,'lstm').eval()
        x = torch.tensor([[2,3,4]]); length = torch.tensor([3])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.pt'
            torch.save(model.state_dict(),path)
            restored = LanguageModel(8,'lstm').eval()
            restored.load_state_dict(torch.load(path,weights_only=True))
            torch.testing.assert_close(model(x,length),restored(x,length))

    def test_full_checkpoint_with_version_metadata(self):
        model = LanguageModel(8,'lstm').eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            torch.save({'state_dict': model.state_dict(), 'config': {
                'vocab_size': 8, 'kind': 'lstm', 'embedding': 48,
                'hidden': 64, 'torch': torch.__version__}},path)
            restored, checkpoint = load_model(path)
            self.assertEqual(str(checkpoint['config']['torch']),str(torch.__version__))
            x = torch.tensor([[2,3,4]]); length = torch.tensor([3])
            torch.testing.assert_close(model(x,length),restored(x,length))

if __name__ == '__main__': unittest.main()
