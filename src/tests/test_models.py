import unittest
import numpy as np

import src.models.atsr_pixel_size as atsr_pixel_size


class MyTestCase(unittest.TestCase):

    def test_compute_pixel_size(self):
        samples = np.array([0, 256, 511])
        target = np.mean([1.16521158, 0.94198235, 1.16521158])
        result = np.mean(atsr_pixel_size.compute()[samples])
        self.assertAlmostEqual(target, result)
