# -*- coding: utf-8 -*-

'''
All the functions that are used to process the global gas flaring
data from ATSR are contained in here.
'''

import os
import logging
import csv
from datetime import datetime

import epr
import numpy as np
from netCDF4 import Dataset
import pyresample as pr
import matplotlib.pyplot as plt
import pandas as pd

import src.config.constants as proc_const
import src.config.filepaths as filepaths


def sun_earth_distance(ats_product):
    doy = datetime.strptime(ats_product.id_string[14:22], "%Y%m%d").timetuple().tm_yday
    return 1 + 0.01672 * np.sin(2 * np.pi * (doy - 93.5) / 365.0)


def radiance_from_reflectance(pixels, ats_product):

    # convert from reflectance to radiance see Smith and Cox 2013
    if 'ATS' in ats_product.id_string:
        solar_irradiance = proc_const.ats_solar_irradiance
    elif 'AT2' in ats_product.id_string:
        solar_irradiance = proc_const.at2_solar_irradiance
    elif 'AT1' in ats_product.id_string:
        solar_irradiance = proc_const.at1_solar_irradiance
    return pixels / 100.0 * solar_irradiance * sun_earth_distance(ats_product) ** 2 / np.pi


def read_atsr(path_to_ats_data):
    return epr.Product(path_to_ats_data)


def read_land_water_mask(path_to_land_water_mask):
    return Dataset(path_to_land_water_mask)


def make_night_mask(ats_product):
    return ats_product.get_band('sun_elev_nadir').read_as_array() < proc_const.day_night_angle


def find_day_geo_subset(day_mask, product):
    '''

    :param day_mask: The daylight mask
    :param product: The atsr product
    :return start_row: The index of where day time starts (to insert the mask back into the full array)
    :return sub_lats: Array subset of lats
    :return sub_lon: Array subset of lons
    '''

    rows, columns = np.where(day_mask == 1)
    start_row = np.min(rows)
    stop_row = np.max(rows)

    sub_lats = product.get_band('latitude').read_as_array(512,
                                                          stop_row - start_row,
                                                          xoffset=0,
                                                          yoffset=start_row)
    sub_lons = product.get_band('longitude').read_as_array(512,
                                                           stop_row - start_row,
                                                           xoffset=0,
                                                           yoffset=start_row)

    return start_row, sub_lats, sub_lons


def setup_water_mask(water_mask_data, sub_lats, sub_lons):

    # Dealing with dateline crossings.  First ATSR observes from East to West
    if (np.min(sub_lons) - -180) < 0.01:
        first_line_max_lon = np.max(sub_lons[0,:])
        min_max_lon = np.min(sub_lons[sub_lons > first_line_max_lon]) # the min of the longitudes that cross dateline

        lon_index_west = int((first_line_max_lon - (-180)) / 360 * water_mask_data['lon'].size)  # lon index
        lon_index_east = int((min_max_lon - (-180)) / 360 * water_mask_data['lon'].size)  # lon index
        #lat_index_north = int((np.max(sub_lats) - 90)*-1 / 180 * water_mask_data['lat'].size) # lat index
        #lat_index_south = int((np.min(sub_lats) - 90)*-1 / 180 * water_mask_data['lat'].size)  # lat index

        # we can run from 0 up to lon_index
        water_mask_west = np.array(water_mask_data['wb_class'][:, 0:lon_index_west])
        water_mask_east = np.array(water_mask_data['wb_class'][:, lon_index_east:])

        # now lets join the masks
        water_mask = np.concatenate((water_mask_east, water_mask_west), axis=1)

        # build the geo grids for the masks
        lons_west = np.tile(water_mask_data['lon'][0:lon_index_west], (water_mask_west.shape[0], 1))
        lons_east = np.tile(water_mask_data['lon'][lon_index_east:], (water_mask_east.shape[0], 1))
        lons = np.concatenate((lons_east, lons_west), axis=1)
        lats = np.transpose(np.tile(water_mask_data['lat'][:],
                                    (water_mask_east.shape[1] + water_mask_west.shape[1], 1)))

    else:
        lon_index_west = int((np.min(sub_lons) - (-180)) / 360 * water_mask_data['lon'].size)  # lon index
        lon_index_east = int((np.max(sub_lons) - (-180)) / 360 * water_mask_data['lon'].size)  # lon index
        #lat_index_north = int((np.max(sub_lats) - 90) * -1 / 180 * water_mask_data['lat'].size)  # lat index
        #lat_index_south = int((np.min(sub_lats) - 90) * -1 / 180 * water_mask_data['lat'].size)  # lat index

        water_mask = np.array(water_mask_data['wb_class'][:, lon_index_west:lon_index_east])
        lons = np.tile(water_mask_data['lon'][lon_index_west:lon_index_east], (water_mask.shape[0], 1))
        lats = np.transpose(np.tile(water_mask_data['lat'][:], (water_mask.shape[1], 1)))

    return water_mask, lons, lats


def make_land_sea_mask():

    # resample land water mask to ATSR grid

    # return where there is water (flag == 2)

    pass


def cloud_mask(ats_product):
    vis_mask = ats_product.get_band('reflec_nadir_0550').read_as_array() > proc_const.vis_thresh
    tir_mask = ats_product.get_band('btemp_nadir_1200').read_as_array() < proc_const.tir_thresh
    return vis_mask | tir_mask


def detect_flares(ats_product, mask):
    swir = ats_product.get_band('reflec_nadir_1600').read_as_array()
    masked_swir = np.ma.masked_array(swir, ~mask)
    return (swir > (masked_swir.mean() + proc_const.n_std * masked_swir.std())) & mask

def mean_background_reflectance(flare_mask, day_sea_mask):
    pass

def compute_frp(pixel_radiances, ats_product):
    if 'ATS' in ats_product.id_string:
        frp_coeff = proc_const.ats_frp_coeff
    elif 'AT2' in ats_product.id_string:
        frp_coeff = proc_const.at2_frp_coeff
    elif 'AT1' in ats_product.id_string:
        frp_coeff = proc_const.at1_frp_coeff
    return proc_const.atsr_pixel_size * frp_coeff * pixel_radiances / 1000000, frp_coeff  # in MW


def flare_data(ats_product, mask):

    # first get data from sensor
    lines, samples = np.where(mask)
    lats = ats_product.get_band('latitude').read_as_array()[mask]
    lons = ats_product.get_band('longitude').read_as_array()[mask]
    reflectances = ats_product.get_band('reflec_nadir_1600').read_as_array()[mask]
    solar_elev_angle = ats_product.get_band('sun_elev_nadir').read_as_array()[mask]
    view_elev_angle = ats_product.get_band('view_elev_nadir').read_as_array()[mask]

    # next radiances from reflectances
    radiances = radiance_from_reflectance(reflectances, ats_product)

    # next get FRP
    frp, frp_coeff = compute_frp(radiances, ats_product)

    # insert data into dataframe
    df = pd.DataFrame()
    datasets = [lines, samples, lats, lons, frp, radiances, reflectances, solar_elev_angle, view_elev_angle]
    names = ['lines', 'samples', 'lats', 'lons', 'frp', 'radiances', 'reflectances', 'sun_elev', 'view_elev', ]
    for k,v in zip(names, datasets):
        df[k] = v
    df['fname'] = ats_product.id_string
    df['se_dist'] = sun_earth_distance(ats_product)
    df['frp_coeff'] = frp_coeff

    # return df
    return df

def save_output():
    f = open(path, 'wb')
    try:
        writer = csv.writer(f)
        writer.writerow(('FRP', 'Radiance', 'Reflectance', 'FRP Coefficient', 'Solar Elevation',
                         'Lat', 'Lon', 'Line', 'Sample', 'View Angle', 'Time', 'Sensor', 'SE_dist'))
    finally:
        f.close()


def ggf_main():
    pass


def main():

    # read in the atsr prodcut and land water
    atsr_fname = 'ATS_TOA_1PUUPA20120406_181820_000065273113_00242_52842_6784.N1'
    atsr_data = read_atsr(filepaths.path_to_aatsr_test_data + atsr_fname)

    # get day/night mask first, we can use this to get only the part of the water mask
    # that we are interested in.  This should massively speed processing.
    night_mask = make_night_mask(atsr_data)

    # get nighttime flare mask
    flare_mask = detect_flares(atsr_data, night_mask)

    # get nighttime flare radiances and frp and write out with meta data
    df = flare_data(atsr_data, flare_mask)








if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()