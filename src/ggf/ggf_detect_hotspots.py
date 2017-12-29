#!/home/users/dnfisher/soft/virtual_envs/ggf/bin/python2

'''
Detects hotspot locations in a given file.  Used later
to detect flares based on hotspot persistency.
'''


import os
import sys
import logging
from datetime import datetime

import epr
import numpy as np
from netCDF4 import Dataset
import pandas as pd

import src.config.constants as proc_const
import src.models.atsr_pixel_size as atsr_pixel_size


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


def detect_hotspots(ats_product):
    swir = ats_product.get_band('reflec_nadir_1600').read_as_array()
    nan_mask = np.isnan(swir)  # get rid of SWIR nans also
    return (swir > proc_const.swir_thresh) & ~nan_mask


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
    sensor = define_sensor(path_to_data)

    # get day/night mask first, we can use this to get only the part of the water mask
    # that we are interested in.  This should massively speed processing.
    night_mask = make_night_mask(atsr_data)

    # get nighttime flare mask
    potential_hotspot_mask = detect_hotspots(atsr_data)
    hotspot_mask = potential_hotspot_mask & night_mask

    # get nighttime flare radiances and frp and write out with meta data
    df = flare_data(atsr_data, hotspot_mask)

    # write out
    output_fname = atsr_data.id_string.split('.')[0] + '_hotspots.csv'
    if sensor.upper() not in output_fname:
        output_fname = output_fname.replace(output_fname[0:3], sensor.upper())
    csv_path = os.path.join(path_to_output, output_fname)
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
