#!/apps/contrib/jaspy/miniconda_envs/jaspy2.7/m2-4.6.14/envs/jaspy2.7-m2-4.6.14-r20190715/bin/python2.7

'''
Detects hotspot locations in a given file.  Used later
to detect flares based on hotspot persistency.
'''


import os
import sys
import logging

import epr
import numpy as np
import pandas as pd

import src.config.constants as proc_const


def read_atsr(path_to_ats_data):
    return epr.Product(path_to_ats_data)


def define_sensor(path_to_data):
    if 'N1' in path_to_data:
        sensor = 'ats'
    if 'E2' in path_to_data:
        sensor = 'at2'
    if 'E1' in path_to_data:
        sensor = 'at1'
    return sensor


def make_night_mask(ats_product):
    solar_elev_angle = np.deg2rad(ats_product.get_band('sun_elev_nadir').read_as_array())
    solar_zenith_angle = np.rad2deg(np.arccos(np.sin(solar_elev_angle)))
    night_mask = solar_zenith_angle >= proc_const.day_night_angle

    swir = ats_product.get_band('reflec_nadir_1600').read_as_array()
    night_swir = swir[night_mask]

    logger.info('Mean nighttime SWIR: ' + str(np.mean(night_swir)))
    logger.info('Median nighttime SWIR: ' + str(np.median(night_swir)))
    logger.info('Max nighttime SWIR: ' + str(np.max(night_swir)))
    logger.info('Min nighttime SWIR: ' + str(np.min(night_swir)))
    logger.info('SD nighttime SWIR: ' + str(np.std(night_swir)))

    return night_mask


def detect_hotspots_fixed(ats_product):
    swir = ats_product.get_band('reflec_nadir_1600').read_as_array()
    nan_mask = np.isnan(swir)  # get rid of SWIR nans also
    return (swir > proc_const.swir_thresh_ats) & ~nan_mask


def detect_hotspots_adaptive(ats_product):

    swir = ats_product.get_band('reflec_nadir_1600').read_as_array()

    # set up the masks
    sza_mask = make_night_mask(ats_product)
    valid_data_mask = ~np.isnan(swir)

    # get threshold
    useable_data = swir[sza_mask & valid_data_mask]
    thresh = np.mean(useable_data) + 4 * np.std(useable_data)
    logger.info('Threshold: ' + str(thresh)) 

    # get all data above threshold
    above_thresh = swir > thresh

    # find flares
    return sza_mask & valid_data_mask & above_thresh


def detect_hotspots_non_parametric(ats_product):

    swir = ats_product.get_band('reflec_nadir_1600').read_as_array()

    # get useful data
    sza_mask = make_night_mask(ats_product)
    valid_data_mask = ~np.isnan(swir)
    useable_data = swir[sza_mask & valid_data_mask]

    # find smallest interval between records for scene
    unique_values = np.unique(useable_data)
    unique_values.sort()
    diff = unique_values[1:] - unique_values[0:-1]
    smallest_diff = np.min(diff)

    # find threshold for data
    useable_data.sort()
    top_subset = useable_data[-5000:]
    diff = top_subset[1:] - top_subset[0:-1]
    diff_mask = diff > smallest_diff
    thresh = np.min(top_subset[1:][diff_mask])
    logger.info('Threshold: ' + str(thresh))

    # get hotspots
    above_thresh = swir > thresh

    return sza_mask & valid_data_mask & above_thresh


def detect_hotspots_min_method(ats_product, sensor='ats'):

    swir = ats_product.get_band('reflec_nadir_1600').read_as_array()

    # get useful data
    sza_mask = make_night_mask(ats_product)
    valid_data_mask = ~np.isnan(swir)
    useable_data = swir[sza_mask & valid_data_mask]

    # if data find minimum
    if useable_data.size:

        if sensor == 'ats':

            min_value = np.min(useable_data)
            thresh = np.abs(min_value)
            logger.info('Threshold: ' + str(thresh))
            above_thresh = swir > thresh

        else:
            above_thresh = swir > 0.06  # assume this is the typical value using AATSR values

        return sza_mask & valid_data_mask & above_thresh
    else:
        logger.info('Threshold not defined - no useable data ')
        return None

def flare_data(product, hotspot_mask):

    # first get data from sensor
    lines, samples = np.where(hotspot_mask)
    lats = product.get_band('latitude').read_as_array()[hotspot_mask]
    lons = product.get_band('longitude').read_as_array()[hotspot_mask]

    # insert data into dataframe
    df = pd.DataFrame()
    datasets = [lines, samples, lats, lons]
    names = ['lines', 'samples', 'lats', 'lons']
    for k,v in zip(names, datasets):
        df[k] = v

    # return df
    return df


def main():

    # read in the atsr product
    path_to_data = sys.argv[1]
    path_to_output = sys.argv[2]
    atsr_data = read_atsr(path_to_data)

    # get day/night mask first
    night_mask = make_night_mask(atsr_data)
    logger.info('Night mask samples ' + str(np.sum(night_mask)))
    logger.info('Night mask shape ' + str(night_mask.shape))

    # get nighttime flare mask
    potential_hotspot_mask = detect_hotspots_fixed(atsr_data)
    logger.info('potential_hotspot_mask samples ' + str(np.sum(potential_hotspot_mask)))
    logger.info('potential_hotspot_mask shape ' + str(potential_hotspot_mask.shape))
    hotspot_mask = potential_hotspot_mask & night_mask
    logger.info('hotspot_mask samples ' + str(np.sum(hotspot_mask)))
    logger.info('hotspot_mask shape ' + str(hotspot_mask.shape))


    # if we exceed this number of flares, then likely
    # something wrong with orbit and reject
    max_flares_in_an_orbit = 20000
    n_flares_detected = np.sum(hotspot_mask)
    if n_flares_detected > max_flares_in_an_orbit:
        logger.info('Too many flares ' + str(n_flares_detected))
        with open(path_to_output, "w"):
            pass
        return

    if hotspot_mask is not None:
        logger.info(path_to_output)
        logger.info('N flares detected: ' + str(n_flares_detected))
        df = flare_data(atsr_data, hotspot_mask)

        # write out
        df.to_csv(path_to_output, index=False)

    else:
        # will fail if no hotspots but still record the processing of the file
        logger.info('No mask, No flares')
        with open(path_to_output, "w"):
            pass
        return


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
