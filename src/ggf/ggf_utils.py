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

import src.config.constants as constants
import src.config.filepaths as filepaths


def sun_earth_distance(doy):
    return 1 + 0.01672 * np.sin(2 * np.pi * (doy - 93.5) / 365.0)


def radiance_from_reflectance():

    # get sun earth distance abd use to compute radiance
    obs_doy = datetime.strptime(product.id_string[14:22], "%Y%m%d").timetuple().tm_yday
    se_dist = sun_earth_distance(obs_doy)

    # convert from reflectance to radiance see Smith and Cox 2013
    data_dict['radiance_16'] = (data_dict['reflectance_16'] /
                                100.0 * data_dict['solar_irradiance']) * se_dist ** 2 / np.pi


def read_atsr():
    ats_product = epr.Product()
    prod_keys = ['reflec_nadir_1600',
                 'btemp_nadir_1200',
                 'latitude',
                 'longitude',
                 'sun_elev_nadir']

    data_dict = {k: product.get_band(v).read_as_array(atsr_pixel_roi['x_samples'],
                                                      atsr_pixel_roi['y_lines'],
                                                      xoffset=atsr_pixel_roi['x_offset'],
                                                      yoffset=atsr_pixel_roi['y_offset'])
                 for k, v in zip(data_keys, prod_keys)}

    return ats_product



def night_day_mask():
    pass


def land_sea_mask():
    pass


def cloud_mask():
    pass


def detect_flares_night():
    pass


def detect_flares_day_sea():
    pass


def compute_frp():
    data_dict['frp'] = ((data_dict['pixel_size'] * constants.sigma / 8.03483163431e-09)
                        * data_dict['radiance_16']) / 1000000  # in MW


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
    pass


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()