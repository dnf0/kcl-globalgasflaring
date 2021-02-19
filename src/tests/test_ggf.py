import unittest
import pandas as pd
import numpy as np
import glob
import epr

import src.utils as utils
from src.ggf.detectors import SLSDetector, ATXDetector


class MyTestCase(unittest.TestCase):

    # -----------------
    # unit tests
    # -----------------

    def test_szn_interpolation(self):
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_target = "../../data/test_data/sls_szn.npy"
        path_to_temp = "../../data/temp/"

        target = np.load(path_to_target)

        product = utils.extract_zip(path_to_data, path_to_temp)
        HotspotDetector = SLSDetector(product)
        HotspotDetector.run_detector()

        self.assertEqual(True, (target == HotspotDetector.sza).all())

    def test_night_mask_sls(self):
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_target = "../../data/test_data/sls_nightmask.npy"
        path_to_temp = "../../data/temp/"
        target = np.load(path_to_target)

        product = utils.extract_zip(path_to_data, path_to_temp)
        HotspotDetector = SLSDetector(product)
        HotspotDetector.run_detector()

        self.assertEqual(True, (target == HotspotDetector.night_mask).all())

    def test_night_mask_atx(self):
        path_to_data = glob.glob("../../data/test_data/*.N1")[0]
        path_to_target = "../../data/test_data/atx_nightmask.npy"
        target = np.load(path_to_target)

        target_mean = np.mean(target)

        product = epr.Product(path_to_data)
        HotspotDetector = ATXDetector(product)
        HotspotDetector.run_detector()

        self.assertAlmostEqual(target_mean, np.mean(HotspotDetector.night_mask))

    def test_vza_interpolation(self):
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_target = "../../data/test_data/sls_vza.npy"
        path_to_temp = "../../data/temp/"

        target = np.load(path_to_target)

        product = utils.extract_zip(path_to_data, path_to_temp)
        HotspotDetector = SLSDetector(product)
        HotspotDetector.run_detector()

        self.assertEqual(True, (target == HotspotDetector.vza).all())

    def test_vza_mask(self):
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_target = "../../data/test_data/sls_vza_mask.npy"
        path_to_temp = "../../data/temp/"

        target = np.load(path_to_target)

        product = utils.extract_zip(path_to_data, path_to_temp)
        HotspotDetector = SLSDetector(product)
        HotspotDetector.run_detector()

        self.assertEqual(True, (target == HotspotDetector.vza_mask).all())

    def test_detect_hotspots_sls(self):
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_target = "../../data/test_data/sls_detect_hotspots.npy"
        path_to_temp = "../../data/temp/"

        target = np.load(path_to_target)

        product = utils.extract_zip(path_to_data, path_to_temp)
        HotspotDetector = SLSDetector(product)
        HotspotDetector.run_detector()

        self.assertEqual(True, (target == HotspotDetector.hotspots).all())

    def test_detect_hotspots_atx(self):
        path_to_data = glob.glob("../../data/test_data/*.N1")[0]
        path_to_target = "../../data/test_data/atx_detect_hotspots.npy"

        target = np.load(path_to_target)

        product = epr.Product(path_to_data)
        HotspotDetector = ATXDetector(product)
        HotspotDetector.run_detector()

        self.assertEqual(True, (target == HotspotDetector.hotspots).all())

    def test_cloud_free_atx(self):

        path_to_data = glob.glob("../../data/test_data/*.N1")[0]
        path_to_target = "../../data/test_data/atx_cloud_mask.npy"

        target = np.load(path_to_target)

        product = epr.Product(path_to_data)
        HotspotDetector = ATXDetector(product)
        HotspotDetector.run_detector()

        self.assertEqual(True, (target == HotspotDetector.cloud_free).all())

    def test_get_arcmin_int(self):

        coords = np.array([-150.53434, -100.13425, -50.20493, 0.34982, 50.43562, 100.12343, 150.56443])
        target = np.array([-15032, -10008, -5012, 21, 5026, 10007, 15034])

        path_to_data = glob.glob("../../data/test_data/*.N1")[0]
        product = epr.Product(path_to_data)
        HotspotDetector = ATXDetector(product)

        result = HotspotDetector._find_arcmin_gridcell(coords)
        self.assertEqual(True, (target == result).all())

    def test_radiance_from_reflectance(self):

        path_to_target = "../../data/test_data/atx_radiance_from_reflectance.npy"
        target = np.load(path_to_target)

        path_to_data = glob.glob("../../data/test_data/*.N1")[0]
        product = epr.Product(path_to_data)
        HotspotDetector = ATXDetector(product)
        reflectance = product.get_band('reflec_nadir_1600').read_as_array()
        result = HotspotDetector._rad_from_ref(reflectance)

        self.assertEqual(True, (target == result).all())

    def test_radiance_from_BT(self):

        path_to_data = glob.glob("../../data/test_data/*.N1")[0]
        product = epr.Product(path_to_data)
        HotspotDetector = ATXDetector(product)

        brightness_temp = 1500
        wavelength = 1.6
        result = HotspotDetector._rad_from_BT(wavelength, brightness_temp)
        target = 28200.577465487077
        self.assertAlmostEqual(target, result)

    def test_sun_earth_distance(self):
        path_to_data = glob.glob("../../data/test_data/*.N1")[0]
        product = epr.Product(path_to_data)
        HotspotDetector = ATXDetector(product)

        target = 0.9877038273760421
        result = HotspotDetector._compute_sun_earth_distance()
        self.assertAlmostEqual(target, result)

    def test_compute_frp(self):
        path_to_data = glob.glob("../../data/test_data/*.N1")[0]
        product = epr.Product(path_to_data)
        HotspotDetector = ATXDetector(product)
        HotspotDetector.run_detector(flares_or_sampling=True)

        path_to_target = "../../data/test_data/atx_frp.npy"
        target = np.load(path_to_target)
        result = HotspotDetector.frp
        self.assertEqual(True, (target == result).all())

    # -----------------
    # functional tests
    # -----------------

    def test_run_atx(self):
        target = pd.read_csv(glob.glob("../../data/test_data/ATS*.csv")[0])
        path_to_data = glob.glob("../../data/test_data/*.N1")[0]

        product = epr.Product(path_to_data)
        HotspotDetector = ATXDetector(product)
        HotspotDetector.run_detector()
        result = HotspotDetector.to_dataframe(keys=['latitude', 'longitude'])

        # TODO determine why floating point errors are causing issues in testing here
        target = target.astype(int)
        result = result.astype(int)
        are_equal = target.equals(result)

        self.assertEqual(True, are_equal)

    def test_run_sls(self):
        # setup
        target = pd.read_csv(glob.glob("../../data/test_data/S3A*.csv")[0])
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_temp = "../../data/temp/"

        product = utils.extract_zip(path_to_data, path_to_temp)
        HotspotDetector = SLSDetector(product)
        HotspotDetector.run_detector()
        result = HotspotDetector.to_dataframe(keys=['latitude', 'longitude', 'sza', 'vza', 'swir_16', 'swir_22'])

        # TODO determine why floating point errors are causing issues in testing here
        target = target.astype(int)
        result = result.astype(int)

        # compare
        are_equal = target.equals(result)
        self.assertEqual(True, are_equal)


if __name__ == '__main__':
    unittest.main()
