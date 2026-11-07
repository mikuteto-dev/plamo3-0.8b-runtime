import struct
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from auxiliary import Indexer, MTP, acceptance, save_weights, select_blocks
from train_auxiliary import qsa_loss


class AuxiliaryTests(unittest.TestCase):
    def test_acceptance_prefix(self):
        self.assertEqual(acceptance([], [5]), 1)
        self.assertEqual(acceptance([7,8], [7,8,9]), 3)
        self.assertEqual(acceptance([7,8], [7,9,10]), 2)
        self.assertEqual(acceptance([7,8], [9,8,10]), 1)
        with self.assertRaises(ValueError):
            acceptance([7,8], [7])

    def test_selection_visible_tail_and_ties(self):
        self.assertEqual(select_blocks([2,2,100], 11, 4), [0,1,2,3,8,9,10])
        self.assertEqual(select_blocks([0,4,2], 12, 8), list(range(4,12)))
        self.assertEqual(select_blocks([], 3, 4), [0,1,2])
        for scores, visible, budget in [([],4,4), ([1],4,5), ([float('nan')],4,4), ([], -1, 4)]:
            with self.assertRaises(ValueError):
                select_blocks(scores, visible, budget)

    def test_indexer_future_is_invisible(self):
        mx.random.seed(42)
        module = Indexer(); module.set_dtype(mx.bfloat16)
        hidden = mx.random.normal((16,1536)).astype(mx.bfloat16)
        positions = mx.array([3,7,11], dtype=mx.int32)
        frequency = 1/(1000000**(mx.arange(64,dtype=mx.float32)/64))
        before = module(hidden, positions, frequency)
        changed = mx.concatenate([hidden[:12], hidden[12:]*100], axis=0)
        after = module(changed, positions, frequency)
        self.assertTrue(bool(mx.all(before == after).item()))
        labels = mx.array([[1,0,0,0],[0.5,0.5,0,0],[1/3,1/3,1/3,0]])
        value, gradients = nn.value_and_grad(module, qsa_loss)(module, hidden, positions, labels, frequency)
        self.assertTrue(bool(mx.isfinite(value).item()))
        self.assertTrue(all(bool(mx.all(mx.isfinite(g)).item()) for _,g in tree_flatten(gradients)))

    def test_mtp_initial_residual_and_export(self):
        predictor = MTP(); predictor.set_dtype(mx.bfloat16)
        x = mx.random.normal((2,3072)).astype(mx.bfloat16)
        out = predictor(x)
        self.assertEqual(out.shape, (2,2,1536))
        self.assertTrue(bool(mx.all(out[:,0] == x[:,:1536]).item()))
        self.assertTrue(bool(mx.all(out[:,1] == x[:,:1536]).item()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.safetensors'
            weights = dict(tree_flatten(predictor.parameters()))
            save_weights(path, weights, {'format':'test'})
            with path.open('rb') as f:
                count = struct.unpack('<Q',f.read(8))[0]
                self.assertEqual((count+8)%64,0)
            restored = mx.load(str(path))
            for name, weight in weights.items():
                self.assertTrue(bool(mx.all(restored[name] == weight).item()))
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                save_weights(path, {'bad': mx.array([float('nan')])}, {})
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
