# -*- coding: utf-8 -*-

'''
All the functions that are used to process the global gas flaring
data from ATSR are contained in here.
'''

import os
import logging

import epr
import numpy as np
from datetime import datetime

import src.config.constants as proc_const
import src.config.filepaths as filepaths


def sun_earth_distance(doy):
    return 1 + 0.01672 * np.sin(2 * np.pi * (doy - 93.5) / 365.0)


def radiance_from_reflectance(pixels, ats_product):

    # get sun earth distance abd use to compute radiance
    obs_doy = datetime.strptime(ats_product.id_string[14:22], "%Y%m%d").timetuple().tm_yday
    sun_earth_dist = sun_earth_distance(obs_doy)

    # convert from reflectance to radiance see Smith and Cox 2013
    return pixels / 100.0 * proc_const.solar_irradiance * sun_earth_dist ** 2 / np.pi


def read_atsr(path_to_ats_data):
    return epr.Product(path_to_ats_data)


def night_day_mask(ats_product):
    return ats_product.get_band('sun_elev_nadir').read_as_array() >= proc_const.day_night_angle


def land_sea_mask():
    pass


def cloud_mask(ats_product):
    vis_mask = ats_product.get_band('reflec_nadir_0550').read_as_array() > proc_const.vis_thresh
    tir_mask = ats_product.get_band('btemp_nadir_1200').read_as_array() < proc_const.tir_thresh
    return vis_mask | tir_mask


def detect_flares(ats_product, mask):
    swir = ats_product.get_band('reflec_nadir_1600').read_as_array()
    masked_swir = np.ma.masked_array(swir, mask)
    return swir > (masked_swir.mean() + proc_const.n_std * masked_swir.std())

def mean_background_reflectance(flare_mask, day_sea_mask):
    pass

def compute_frp(pixel_radiances):
    return proc_const.atsr_pixel_size * proc_const.frp_coeff * pixel_radiances / 1000000  # in MW


def save_output():
    f = open(path, 'wb')
    try:
        writer = csv.writer(f)
        writer.writerow(('Power', "Radiance", 'BT',
                         'Lat', "Lon", 'Scan Angle', 'Time'))
    finally:
        f.close()


def ggf_main():
    pass


def main():

    # read in the atsr prodcut

    # get day/night mask

    # get land/ sea and cloud mask and combine both with day mask

    # get nighttime flare mask

    # get daytime flare mask

    # get mean background reflectance information for daytime flares

    # get nighttime flare radiances and frp and write out with meta data

    # get daytime flare radiances and frp and write our with meta data








if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()