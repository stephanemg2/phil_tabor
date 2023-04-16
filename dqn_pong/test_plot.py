import unittest
import matplotlib.pyplot as plt
import numpy as np
import os

from utils import plot_learning_curve


class TestPlotLearningCurve(unittest.TestCase):

    def setUp(self):
        self.x = np.arange(100)
        self.scores = np.random.rand(100)
        self.epsilons = np.random.rand(100)
        self.filename = "test.png"

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_plot_learning_curve(self):
        plot_learning_curve(self.x, self.scores, self.epsilons, self.filename)
        self.assertTrue(os.path.exists(self.filename))


if __name__ == '__main__':
    unittest.main()