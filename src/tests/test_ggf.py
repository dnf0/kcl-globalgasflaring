import unittest
import pandas as pd
import numpy as np
import glob
import os

import src.ggf.ggf_extract_flares_and_samples_atx as ggf_extract_flares_and_samples_atx
import src.utils as utils
import src.config.constants as proc_const
from src.ggf.detectors import SLSDetector, ATXDetector


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

        result = utils.extract_zip(path_to_data, path_to_temp)
        self.assertEqual(target.keys(), result.keys())

    def test_szn_interpolation(self):
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_target = "../../data/test_data/sls_szn.npy"
        path_to_temp = "../../data/temp/"

        target = np.load(path_to_target)

        product = utils.extract_zip(path_to_data, path_to_temp)
        HotspotDetector = SLSDetector(proc_const.day_night_angle,
                                      proc_const.s5_rad_thresh,
                                      product)
        HotspotDetector.run_detector()

        self.assertEqual(True, (target == HotspotDetector.sza).all())

    def test_night_mask_sls(self):
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_target = "../../data/test_data/sls_nightmask.npy"
        path_to_temp = "../../data/temp/"
        target = np.load(path_to_target)

        product = utils.extract_zip(path_to_data, path_to_temp)
        HotspotDetector = SLSDetector(proc_const.day_night_angle,
                                      proc_const.s5_rad_thresh,
                                      product)
        HotspotDetector.run_detector()

        self.assertEqual(True, (target == HotspotDetector.night_mask).all())

    def test_night_mask_atx(self):
        path_to_data = glob.glob("../../data/test_data/*.N1")[0]
        path_to_target = "../../data/test_data/atx_nightmask.npy"
        target = np.load(path_to_target)

        target_mean = np.mean(target)

        product = utils.read_atsr(path_to_data)
        HotspotDetector = ATXDetector(proc_const.day_night_angle,
                                      proc_const.swir_thresh_ats,
                                      product)
        HotspotDetector.run_detector()

        self.assertAlmostEqual(target_mean, np.mean(HotspotDetector.night_mask))

    def test_vza_interpolation(self):
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_target = "../../data/test_data/sls_vza.npy"
        path_to_temp = "../../data/temp/"

        target = np.load(path_to_target)

        product = utils.extract_zip(path_to_data, path_to_temp)
        HotspotDetector = SLSDetector(proc_const.day_night_angle,
                                      proc_const.s5_rad_thresh,
                                      product)
        HotspotDetector.run_detector()

        self.assertEqual(True, (target == HotspotDetector.vza).all())

    def test_vza_mask(self):
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_target = "../../data/test_data/sls_vza_mask.npy"
        path_to_temp = "../../data/temp/"

        target = np.load(path_to_target)

        product = utils.extract_zip(path_to_data, path_to_temp)
        HotspotDetector = SLSDetector(proc_const.day_night_angle,
                                      proc_const.s5_rad_thresh,
                                      product)
        HotspotDetector.run_detector()

        self.assertEqual(True, (target == HotspotDetector.vza_mask).all())

    def test_detect_hotspots_sls(self):
        path_to_data = glob.glob("../../data/test_data/S3A*.zip")[0]
        path_to_target = "../../data/test_data/sls_detect_hotspots.npy"
        path_to_temp = "../../data/temp/"

        target = np.load(path_to_target)

        product = utils.extract_zip(path_to_data, path_to_temp)
        HotspotDetector = SLSDetector(proc_const.day_night_angle,
                                      proc_const.s5_rad_thresh,
                                      product)
        HotspotDetector.run_detector()

        self.assertEqual(True, (target == HotspotDetector.potential_hotspots).all())

    def test_detect_hotspots_atx(self):
        path_to_data = glob.glob("../../data/test_data/*.N1")[0]
        path_to_target = "../../data/test_data/atx_detect_hotspots.npy"

        target = np.load(path_to_target)

        product = utils.read_atsr(path_to_data)
        HotspotDetector = ATXDetector(proc_const.day_night_angle,
                                      proc_const.swir_thresh_ats,
                                      product)
        HotspotDetector.run_detector()

        self.assertEqual(True, (target == HotspotDetector.potential_hotspots).all())

    def test_make_cloud_mask_atx(self):

        path_to_data = glob.glob("../../data/test_data/*.N1")[0]
        path_to_target = "../../data/test_data/atx_cloud_mask.npy"

        target = np.load(path_to_target)

        atx_data = ggf_extract_hotspots_atx.read_atsr(path_to_data)
        result = ggf_extract_flares_and_samples_atx.make_cloud_mask(atx_data)

        self.assertEqual(True, (target == result).all())

    def test_get_arcmin_int(self):

        coords = np.array([-150.53434, -100.13425, -50.20493, 0.34982, 50.43562, 100.12343, 150.56443])
        target = np.array([-15032, -10008, -5012, 21, 5026, 10007, 15034])
        result = ggf_extract_flares_and_samples_atx.get_arcmin_int(coords)
        self.assertEqual(True, (target == result).all())

    def test_round_to_arcmin(self):

        coord = 12.1234
        target = 12.116666666666667
        result = ggf_extract_flares_and_samples_atx.round_to_arcmin(coord)
        self.assertAlmostEqual(target, result)

    def test_radiance_from_reflectance(self):
        path_to_data = glob.glob("../../data/test_data/*.N1")[0]
        path_to_target = "../../data/test_data/atx_radiance_from_reflectance.npy"

        target = np.load(path_to_target)

        atx_data = ggf_extract_hotspots_atx.read_atsr(path_to_data)
        sensor = 'ats'

        swir_reflectances = atx_data.get_band('reflec_nadir_1600').read_as_array()[:]
        result = ggf_extract_flares_and_samples_atx.radiance_from_reflectance(swir_reflectances,
                                                                                      atx_data,
                                                                                      sensor)
        self.assertEqual(True, (target == result).all())

    def test_radiance_from_BT(self):

        brightness_temp = 1500
        wavelength = 1.6
        result = ggf_extract_flares_and_samples_atx.radiance_from_BT(wavelength, brightness_temp)
        target = 28200.577465487077
        self.assertAlmostEqual(target, result)

    def test_sun_earth_distance(self):
        id_string = "ATS_TOA_1PUUPA20101120_153455_000065273096_00341_45614_9478.N1"
        target = 0.9877038273760421
        result = ggf_extract_flares_and_samples_atx.sun_earth_distance(id_string)
        self.assertAlmostEqual(target, result)

    def test_compute_pixel_size(self):
        samples = np.array([0, 256, 511])
        target = np.mean([1165211.58382742, 941982.35386131, 1165211.58382712])
        result = np.mean(ggf_extract_flares_and_samples_atx.compute_pixel_size(samples))
        self.assertAlmostEqual(target, result)

    def test_compute_frp(self):
        pixel_radiances = np.array([2, 5, 3])
        pixel_sizes = np.array([1165211.58382742, 941982.35386131, 1165211.58382712])
        sensor = 'ats'
        target = np.mean([15.92267048, 32.18058168, 23.88400572])
        result = np.mean(ggf_extract_flares_and_samples_atx.compute_frp(pixel_radiances, pixel_sizes, sensor))
        self.assertAlmostEqual(target, result, places=2)


    def test_construct_hotspot_line_sample_df(self):
        path_to_data = glob.glob("../../data/test_data/*.N1")[0]
        atx_data = ggf_extract_hotspots_atx.read_atsr(path_to_data)

        path_to_data = glob.glob("../../data/test_data/atx_detect_hotspots.npy")[0]
        hotspot_mask = np.load(path_to_data)

        path_to_target = "../../data/test_data/atx_line_sample_df.csv"
        target = pd.read_csv(path_to_target, index_col=[0])

        result = ggf_extract_flares_and_samples_atx.construct_hotspot_line_sample_df(atx_data, hotspot_mask)

        # compare
        are_equal = result.equals(target)
        self.assertEqual(True, are_equal)

    # -----------------
    # functional tests
    # -----------------

    def test_run_atx(self):
        target = pd.read_csv(glob.glob("../../data/test_data/ATS*.csv")[0])
        path_to_data = glob.glob("../../data/test_data/*.N1")[0]
        path_to_output = "../../data/test_data/ggf_extract_hotspots_atx_test_result.csv"
        if os.path.exists(path_to_output):
            os.remove(path_to_output)

        # call
        ggf_extract_hotspots_atx.run(path_to_data, path_to_output)

        # compare (to two decimal places)
        result = pd.read_csv(path_to_output)
        target = target.round(2)
        result = result.round(2)
        are_equal = target.equals(result)
        self.assertEqual(True, are_equal)

    def test_run_sls(self):
        # setup
        target = pd.read_csv(glob.glob("../../data/test_data/S3A*.csv")[0])
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
