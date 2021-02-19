import unittest
import glob
import src.utils as utils


class MyTestCase(unittest.TestCase):

    def test_extract_zip(self):
        target = {"S5_radiance_an": None,
                  "S6_radiance_an": None,
                  "geodetic_an": None,
                  "geometry_tn": None,
                  "cartesian_an": None,
                  "cartesian_tx": None,
                  "indices_an": None,
                  "flags_an": None,
                  'time_an': None}
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_temp = "../../data/temp/"

        result = utils.extract_zip(path_to_data, path_to_temp)
        self.assertEqual(target.keys(), result.keys())

