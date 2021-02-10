import unittest
import pandas as pd
import numpy as np
import pickle
import glob
import os

import src.ggf.ggf_extract_hotspots_sls as ggf_extract_hotspots_sls


class MyTestCase(unittest.TestCase):
    # -----------------
    # unit tests
    # -----------------

    def test_extract_zip(self):
        target = {"S5_radiance_an": None,
                  "S6_radiance_an": None,
                  "geodetic_an": None,
                  "geometry_tn": None,
                  "cartesian_an": None,
                  "cartesian_tx": None,
                  "indices_an": None,
                  "flags_an": None}
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_temp = "../../data/temp/"

        result = ggf_extract_hotspots_sls.extract_zip(path_to_data, path_to_temp)
        self.assertEqual(target.viewkeys(), result.viewkeys())

    def test_szn_interpolation(self):
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_target = "../../data/test_data/sls_szn.npy"
        path_to_temp = "../../data/temp/"

        target = np.load(path_to_target)

        s3_data = ggf_extract_hotspots_sls.extract_zip(path_to_data, path_to_temp)
        result = ggf_extract_hotspots_sls.interpolate_szn(s3_data)

        self.assertEqual(True, (target == result).all())

    def test_night_mask(self):
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_target = "../../data/test_data/sls_nightmask.npy"
        path_to_temp = "../../data/temp/"
        target = np.load(path_to_target)

        s3_data = ggf_extract_hotspots_sls.extract_zip(path_to_data, path_to_temp)
        sza, result = ggf_extract_hotspots_sls.make_night_mask(s3_data)

        self.assertEqual(True, (target == result).all())

    def test_vza_interpolation(self):
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_target = "../../data/test_data/sls_vza.npy"
        path_to_temp = "../../data/temp/"

        target = np.load(path_to_target)

        s3_data = ggf_extract_hotspots_sls.extract_zip(path_to_data, path_to_temp)
        result = ggf_extract_hotspots_sls.interpolate_vza(s3_data)

        self.assertEqual(True, (target == result).all())

    def test_vza_mask(self):
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_target = "../../data/test_data/sls_vza_mask.npy"
        path_to_temp = "../../data/temp/"

        target = np.load(path_to_target)

        s3_data = ggf_extract_hotspots_sls.extract_zip(path_to_data, path_to_temp)
        vza, result = ggf_extract_hotspots_sls.make_vza_mask(s3_data)

        self.assertEqual(True, (target == result).all())

    def test_detect_hotspots(self):
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_target = "../../data/test_data/detect_hotspots.npy"
        path_to_temp = "../../data/temp/"

        target = np.load(path_to_target)

        s3_data = ggf_extract_hotspots_sls.extract_zip(path_to_data, path_to_temp)
        result = ggf_extract_hotspots_sls.detect_hotspots(s3_data)

        self.assertEqual(True, (target == result).all())

    # -----------------
    # functional tests
    # -----------------

    def test_run_sls(self):
        # setup
        target = pd.read_csv(glob.glob("../../data/test_data/S3A*test.csv")[0])
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_temp = "../../data/temp/"
        path_to_output = "../../data/test_data/ggf_extract_hotspots_sls_test_result.csv"
        if os.path.exists(path_to_output):
            os.remove(path_to_output)

        # call
        ggf_extract_hotspots_sls.run(path_to_data, path_to_temp, path_to_output)

        # compare
        result = pd.read_csv(path_to_output)
        are_equal = target.equals(result)
        self.assertEqual(True, are_equal)


if __name__ == '__main__':
    unittest.main()
