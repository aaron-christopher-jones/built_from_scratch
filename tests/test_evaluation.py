# test_evaluation.py

import numpy as np
import unittest

from scratch.utils import evaluation as eval


class TestEvaluation(unittest.TestCase):
    def test_metric_rmse(self):
        y = np.array([1.2, 7.9, 4.6, 2.2])
        yhat = np.array([2.1, 6.5, 4.4, 2.7])
        self.assertEqual(eval.metric_rmse(y, yhat), 0.8746)

    def test_metric_maep(self):
        y = np.array([1.2, 7.9, 4.6, 2.2])
        yhat = np.array([2.1, 6.5, 4.4, 2.7])
        self.assertEqual(eval.metric_maep(y, yhat), 0.1887)


if __name__ == '__main__':
    unittest.main()
