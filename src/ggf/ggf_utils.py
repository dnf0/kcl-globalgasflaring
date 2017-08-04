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
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

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
        lat_index_north = int((np.max(sub_lats) - 90)*-1 / 180 * water_mask_data['lat'].size) # lat index
        lat_index_south = int((np.min(sub_lats) - 90)*-1 / 180 * water_mask_data['lat'].size)  # lat index

        # we can run from 0 up to lon_index
        print 'start read'
        water_mask_west = water_mask_data['wb_class'][lat_index_north:lat_index_south, 0:lon_index_west]
        water_mask_east = water_mask_data['wb_class'][lat_index_north:lat_index_south, lon_index_east:]
        print 'done read'


    else:
        pass


    # cols = np.where((water_mask_data['lon'][:] >= min_lon) & (water_mask_data['lon'][:] <= max_lon))
    # rows = np.where((water_mask_data['lat'][:] >= min_lat) & (water_mask_data['lat'][:] <= max_lat))
    #
    # min_row = np.min(rows) - 5
    # max_row = np.max(rows) + 5
    # min_col = np.min(cols) - 5
    # max_col = np.max(cols) + 5
    #
    # water_mask_subset = water_mask_data['wb_class'][min_row:max_row, min_col,max_col]
    #
    # lats = water_mask_data['lat'][:]
    # lons = water_mask_data['lon'][:]


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

    # read in the atsr prodcut and land water
    atsr_fname = 'ATS_TOA_1PUUPA20120406_181820_000065273113_00242_52842_6784.N1'
    atsr_data = read_atsr(filepaths.path_to_aatsr_test_data + atsr_fname)
    water_mask_data = read_land_water_mask(filepaths.path_to_landcover_test)

    # get day/night mask first, we can use this to get only the part of the water mask
    # that we are interested in.  This should massively speed processing.
    night_mask = make_night_mask(atsr_data)
    day_mask = ~night_mask

    # find the grids of the subset and get where they start in the array
    start_row, lat_subset, lon_subset = find_day_geo_subset(day_mask, atsr_data)

    # set up water mask
    water_mask, water_lat, water_lon = setup_water_mask(water_mask_data, lat_subset, lon_subset)

    # set up pyresample geographic grids
    water_mask_def = pr.geometry.GridDefinition(lons=atsr_data.get_band('longitude').read_as_array(),
                                                lats=atsr_data.get_band('latitude').read_as_array())
    atsr_swath_def = pr.geometry.SwathDefinition(lons=atsr_data.get_band('longitude').read_as_array(),
                                                 lats=atsr_data.get_band('latitude').read_as_array())




    # get land/water and cloud mask and combine both with day mask

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