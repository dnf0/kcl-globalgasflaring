#!/home/users/dnfisher/soft/virtual_envs/ggf/bin/python2

'''
Detects hotspot locations in a given file.  Used later
to detect flares based on hotspot persistency.
'''


import sys
import os
import logging
import zipfile
import shutil
import glob

import numpy as np
from netCDF4 import Dataset
import pandas as pd
from scipy.interpolate import RectBivariateSpline

import src.config.constants as proc_const


def extract_zip(input_zip, path_to_temp):
    data_dict = {}
    to_extract = ["S5_radiance_an.nc", "S6_radiance_an.nc",
                  "geodetic_an.nc", "geometry_tn.nc",
                  "cartesian_an.nc", "cartesian_tx.nc",
                  "indices_an.nc", "flags_an.nc"]
    with zipfile.ZipFile(input_zip) as input:
        for name in input.namelist():
            split_name = name.split('/')[-1]
            if split_name in to_extract:
                var_name = split_name.split('.')[0]
                source = Dataset(input.extract(name, path_to_temp))
                data_dict[var_name] = source

    # remove the unzip files
    dir_to_remove = os.path.join(path_to_temp, input_zip.split('/')[-1].replace('zip', 'SEN3'))
    if os.path.isdir(dir_to_remove):  # test if the path points to a directory
        shutil.rmtree(dir_to_remove, ignore_errors=True)
    else:  # normal file
        os.remove(dir_to_remove)

    return data_dict


def interpolate_szn(s3_data):
    szn = s3_data['geometry_tn']['solar_zenith_tn'][:]

    tx_x_var = s3_data['cartesian_tx']['x_tx'][0, :]
    tx_y_var = s3_data['cartesian_tx']['y_tx'][:, 0]

    an_x_var = s3_data['cartesian_an']['x_an'][:]
    an_y_var = s3_data['cartesian_an']['y_an'][:]

    spl = RectBivariateSpline(tx_y_var, tx_x_var[::-1], szn[:, ::-1].filled(0))
    interpolated = spl.ev(an_y_var.compressed(),
                          an_x_var.compressed())
    interpolated = np.ma.masked_invalid(interpolated, copy=False)
    szn = np.ma.empty(an_y_var.shape,
                      dtype=szn.dtype)
    szn[np.logical_not(np.ma.getmaskarray(an_y_var))] = interpolated
    szn.mask = an_y_var.mask
    return szn


def make_night_mask(s3_data):
    solar_zenith_angle = interpolate_szn(s3_data)
    return solar_zenith_angle, solar_zenith_angle.filled(0) >= proc_const.day_night_angle


def interpolate_vza(s3_data):
    sat_zn = s3_data['geometry_tn']['sat_zenith_tn'][:]

    tx_x_var = s3_data['cartesian_tx']['x_tx'][0, :]
    tx_y_var = s3_data['cartesian_tx']['y_tx'][:, 0]

    an_x_var = s3_data['cartesian_an']['x_an'][:]
    an_y_var = s3_data['cartesian_an']['y_an'][:]

    spl = RectBivariateSpline(tx_y_var, tx_x_var[::-1], sat_zn[:, ::-1].filled(0))
    interpolated = spl.ev(an_y_var.compressed(),
                          an_x_var.compressed())
    interpolated = np.ma.masked_invalid(interpolated, copy=False)
    sat_zn = np.ma.empty(an_y_var.shape,
                      dtype=sat_zn.dtype)
    sat_zn[np.logical_not(np.ma.getmaskarray(an_y_var))] = interpolated
    sat_zn.mask = an_y_var.mask
    return sat_zn


def make_vza_mask(s3_data):
    view_zenith_angles = interpolate_vza(s3_data)
    return view_zenith_angles, view_zenith_angles.filled(100) <= 22


def detect_hotspots(s3_data):
    # fill nan's with zero.  Solar constant comes from SLSTR viscal product
    thresh = proc_const.swir_thresh_sls / (100 * np.pi) * 254.23103333  # convert ref threhsold to rad
    return s3_data['S5_radiance_an']['S5_radiance_an'][:].filled(0) > thresh


def detect_hotspots_adaptive(ds, sza_mask, vza_mask):

    # first get unillimunated central swath data
    valid_mask = ds != -999
    useable_data = ds[sza_mask & vza_mask & valid_mask]

    # get threshold
    thresh = np.mean(useable_data) + 4 * np.std(useable_data)
    logger.info('Threshold: ' + str(thresh))


    # get all data above threshold
    above_thresh = ds > thresh

    # find flares
    return sza_mask & vza_mask & valid_mask & above_thresh


def detect_hotspots_non_parametric(ds, sza_mask, vza_mask):

    # first get unillimunated central swath data
    valid_mask = ds != -999
    useable_data = ds[sza_mask & vza_mask & valid_mask]

    # find threshold for data
    useable_data.sort()
    top_subset = useable_data[-1000:]
    bottom_subset = useable_data[:1000]

    diff_top = top_subset[1:] - top_subset[0:-1]
    diff_bottom = bottom_subset[1:] - bottom_subset[0:-1]

    # check the max differences for both top and bottom 1k radiances
    # if they are similar - within 5 DNs, then assume no flares.  This
    # is reasonable as flares increase as step functions (with changes likely
    # to be greater than a couple of DN points).  This helps remove scenes
    # with no flaring activity in them.  Otherwise, get hundreds of invalid
    # datapoints
    max_diff_top = np.max(diff_top)
    max_diff_bottom = np.max(diff_bottom)

    logger.info('Max diff top 1k: ' + str(max_diff_top))
    logger.info('Max diff bottom 1k: ' + str(max_diff_bottom))

    if max_diff_top <= 5*max_diff_bottom:
        return None

    # now get the smallest non-zero difference for the top 1k values
    not_zero = diff_top != 0
    smallest_diff = np.min(diff_top[not_zero])

    logger.info('smallest diff for top 1k : ' + str(smallest_diff))

    diff_mask = diff_top > smallest_diff
    thresh = np.min(top_subset[1:][diff_mask])

    logger.info('Threshold using scene smallest diff: ' + str(thresh))

    # get hotspots
    above_thresh = ds > thresh

    return sza_mask & vza_mask & valid_mask & above_thresh


def detect_hotspots_min_method(ds, sza_mask, vza_mask):

    # first get unillimunated central swath data
    valid_mask = ds != -999
    useable_data = ds[sza_mask & vza_mask & valid_mask]

    # if data find minimum
    if useable_data.size:

        min_value = np.min(useable_data)
        thresh = np.abs(min_value)
        logger.info('Threshold: ' + str(thresh))
        above_thresh = ds > thresh

        return sza_mask & vza_mask & valid_mask & above_thresh
    else:
        logger.info('Threshold not defined - no useable data ')
        return None



def flare_data(s3_data, sza, vza, hotspot_mask):

    lines, samples = np.where(hotspot_mask)
    lats = s3_data['geodetic_an']['latitude_an'][:][hotspot_mask]
    lons = s3_data['geodetic_an']['longitude_an'][:][hotspot_mask]
    sza = sza[hotspot_mask]
    vza = vza[hotspot_mask]
    S5 = s3_data['S5_radiance_an']['S5_radiance_an'][:][hotspot_mask]
    S6 = s3_data['S6_radiance_an']['S6_radiance_an'][:][hotspot_mask]

    df = pd.DataFrame()
    datasets = [lines, samples, lats, lons, sza, vza, S5, S6]
    names = ['lines', 'samples', 'lats', 'lons', 'sza', 'vza', 'S5', 'S6']
    for k,v in zip(names, datasets):
        df[k] = v

    return df


def main():

    path_to_data = sys.argv[1]
    path_to_output = sys.argv[2]
    path_to_temp = sys.argv[3]

    s3_data = extract_zip(path_to_data, path_to_temp)

    # load in S5 and S6 channels
    s5_data = s3_data['S5_radiance_an']['S5_radiance_an'][:].filled(-999)

    # get vza and sza masks
    try:
        vza, vza_mask = make_vza_mask(s3_data)
        sza, sza_mask = make_night_mask(s3_data)
    except Exception, e:
        logger.info('Could not generate angular data with error: ' + str(e))
        with open(path_to_output, "w"):
            pass
        return

    if sza_mask.max() == 0:  # all daytime data
        logger.info('No flares detected: Illuminated scene')
        with open(path_to_output, "w"):
            pass
        return

    # get the hotspot data for both channels and then generate the mask
    try:
        hotspot_mask = detect_hotspots_min_method(s5_data, sza_mask, vza_mask)
    except Exception, e:
        logger.warning('Could not genrate hotspot mask with error: ' + str(e))
        with open(path_to_output, "w"):
            pass
        return

    if hotspot_mask is not None:
        logger.info(path_to_output)
        logger.info('N flares detected: ' + str(np.sum(hotspot_mask)))
        df = flare_data(s3_data, sza, vza, hotspot_mask)

        # write out
        df.to_csv(path_to_output, index=False)
        return

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
