#!/home/users/dnfisher/soft/virtual_envs/ggf/bin/python2

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
    return solar_zenith_angle >= proc_const.day_night_angle


def detect_hotspots_fixed(swir):
    return swir > proc_const.swir_thresh_ats


def detect_hotspots_adaptive(ats_product):

    swir = ats_product.get_band('reflec_nadir_1600').read_as_array()

    # set up the masks
    sza_mask = make_night_mask(ats_product)
    valid_data_mask = ~np.isnan(swir)

    # get threshold
    useable_data = swir[sza_mask & valid_data_mask]
    thresh = np.mean(useable_data) + 4 * np.std(useable_data)

    # get all data above threshold
    above_thresh = swir > thresh

    # find flares
    return sza_mask & valid_data_mask & above_thresh


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

    # read in the atsr prodcut and land water
    path_to_data = sys.argv[1]
    path_to_output = sys.argv[2]
    atsr_data = read_atsr(path_to_data)

    # get nighttime flare mask
    hotspot_mask = detect_hotspots_adaptive(atsr_data)
    logger.info(path_to_output)
    logger.info('N flares detected: ' + str(np.sum(hotspot_mask)))

    # get nighttime flare radiances and frp and write out with meta data
    flare_count_limit = 5000  # should not be morethan this many flares in one orbit (gets rid of dodgy AT2 data)!
    if np.sum(hotspot_mask) > flare_count_limit:
        with open(path_to_output, "w"):
            pass
    else:
        df = flare_data(atsr_data, hotspot_mask)
        
        # write out
        df.to_csv(path_to_output, index=False)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
