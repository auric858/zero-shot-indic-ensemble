import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import ensemble as E


def onehot(preds, C=3, p=0.9):
    """Fake model output: probability p on the chosen class, the rest spread evenly."""
    out = np.full((len(preds), C), (1 - p) / (C - 1))
    out[np.arange(len(preds)), preds] = p
    return out


class TestEnsemble(unittest.TestCase):
    def test_majority_recovery(self):
        truth = np.array([0, 1, 2, 1, 0])
        probs = np.stack([onehot(truth), onehot(truth), onehot((truth + 1) % 3)])
        self.assertTrue((E.hard_vote(probs) == truth).all())
        self.assertTrue((E.soft_vote(probs) == truth).all())

    def test_hard_vote_majority(self):
        probs = np.stack([onehot([0]), onehot([0]), onehot([1], p=0.99)])
        self.assertEqual(E.hard_vote(probs)[0], 0)

    def test_tie_break_by_confidence(self):
        probs = np.stack([onehot([0], p=0.6), onehot([1], p=0.95)])
        self.assertEqual(E.hard_vote(probs)[0], 1)

    def test_weight_handling(self):
        probs = np.stack([onehot([0]), onehot([1])])
        self.assertEqual(E.weighted_soft_vote(probs, [0.9, 0.1])[0], 0)
        self.assertEqual(E.weighted_soft_vote(probs, [0.1, 0.9])[0], 1)

    def test_probability_normalization(self):
        probs = np.stack([onehot([0, 1]), onehot([1, 1])])
        self.assertTrue(np.allclose(E.weighted_soft_vote_probs(probs, [3, 1]).sum(1), 1))
        self.assertTrue(np.allclose(E.soft_vote_probs(probs).sum(1), 1))

    def test_invalid_weights(self):
        probs = np.stack([onehot([0]), onehot([1])])
        for bad in ([-1, 2], [0, 0], [1], [np.nan, 1]):
            with self.assertRaises(ValueError):
                E.weighted_soft_vote(probs, bad)


if __name__ == "__main__":
    unittest.main()
    