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

import numpy as np
from netCDF4 import Dataset
import pandas as pd
from scipy.interpolate import RectBivariateSpline

import src.config.constants as proc_const


def extract_zip(input_zip, path_to_temp):
    data_dict = {}
    to_extract = ["S5_radiance_cn.nc",
                  "geodetic_cn.nc", "geometry_tn.nc",
                  "cartesian_cn.nc", "cartesian_tx.nc",
                  "indices_cn.nc", "flags_cn.nc"]
    with zipfile.ZipFile(input_zip) as input:
        for name in input.namelist():
            split_name = name.split('/')[-1]
            if split_name in to_extract:
                var_name = split_name.split('.')[0]
                source = Dataset(input.extract(name, path_to_temp))
                data_dict[var_name] = source

    # remove the unzip files
    extracted_files = os.listdir(".")
    for f in extracted_files:
        absolute_path = os.path.abspath(f)  # get the absolute path
        if os.path.isdir(absolute_path):  # test if the path points to a directory
            shutil.rmtree(absolute_path)
        else:  # normal file
            os.remove(absolute_path)

    return data_dict


def interpolate_szn(s3_data):
    szn = s3_data['geometry_tn']['solar_zenith_tn'][:]

    tx_x_var = s3_data['cartesian_tx']['x_tx'][0, :]
    tx_y_var = s3_data['cartesian_tx']['y_tx'][:, 0]

    cn_x_var = s3_data['cartesian_cn']['x_cn'][:]
    cn_y_var = s3_data['cartesian_cn']['y_cn'][:]

    spl = RectBivariateSpline(tx_y_var, tx_x_var[::-1], szn[:, ::-1].filled(0))
    interpolated = spl.ev(cn_y_var.compressed(),
                          cn_x_var.compressed())
    interpolated = np.ma.masked_invalid(interpolated, copy=False)
    szn = np.ma.empty(cn_y_var.shape,
                      dtype=szn.dtype)
    szn[np.logical_not(np.ma.getmaskarray(cn_y_var))] = interpolated
    szn.mask = cn_y_var.mask
    return szn


def make_night_mask(s3_data):
    solar_zenith_angle = interpolate_szn(s3_data)
    return solar_zenith_angle.filled(0) >= proc_const.day_night_angle


def interpolate_vza(s3_data):
    sat_zn = s3_data['geometry_tn']['sat_zenith_tn'][:]

    tx_x_var = s3_data['cartesian_tx']['x_tx'][0, :]
    tx_y_var = s3_data['cartesian_tx']['y_tx'][:, 0]

    cn_x_var = s3_data['cartesian_cn']['x_cn'][:]
    cn_y_var = s3_data['cartesian_cn']['y_cn'][:]

    spl = RectBivariateSpline(tx_y_var, tx_x_var[::-1], sat_zn[:, ::-1].filled(0))
    interpolated = spl.ev(cn_y_var.compressed(),
                          cn_x_var.compressed())
    interpolated = np.ma.masked_invalid(interpolated, copy=False)
    sat_zn = np.ma.empty(cn_y_var.shape,
                      dtype=sat_zn.dtype)
    sat_zn[np.logical_not(np.ma.getmaskarray(cn_y_var))] = interpolated
    sat_zn.mask = cn_y_var.mask
    return sat_zn


def make_vza_mask(s3_data):
    view_zenith_angles = interpolate_vza(s3_data)
    return view_zenith_angles.filled(100) <= 22


def detect_hotspots(s3_data):
    # fill nan's with zero.  Solar constant comes from SLSTR viscal product
    swir = s3_data['S5_radiance_cn']['S5_radiance_cn'][:].filled(0) / 254.23103333 * np.pi * 100
    return swir > proc_const.swir_thresh


def flare_data(s3_data, hotspot_mask):

    lines, samples = np.where(hotspot_mask)
    lats = s3_data['geodetic_cn']['latitude_cn'][:][hotspot_mask]
    lons = s3_data['geodetic_cn']['longitude_cn'][:][hotspot_mask]

    df = pd.DataFrame()
    datasets = [lines, samples, lats, lons]
    names = ['lines', 'samples', 'lats', 'lons']
    for k,v in zip(names, datasets):
        df[k] = v

    return df


def main():

    # read in the atsr prodcut and land water
    path_to_data = sys.argv[1]
    path_to_output = sys.argv[2]
    path_to_temp = sys.argv[3]

    # open zip file
    s3_data = extract_zip(path_to_data, path_to_temp)

    # get day/night mask, if no night mask return save empty df
    night_mask = make_night_mask(s3_data)
    if night_mask.max() == 0:
        return

    # set up the vza mask so that we only get data similar to atsr
    vza_mask = make_vza_mask(s3_data)

    # get nighttime hotspots
    potential_hotspot_mask = detect_hotspots(s3_data)
    hotspot_mask = potential_hotspot_mask & night_mask & vza_mask

    # get nighttime flare radiances and frp and write out with meta data
    df = flare_data(s3_data, hotspot_mask)

    # write out
    output_fname = path_to_data.split('/')[-1].split('.')[0] + '_hotspots.csv'
    csv_path = os.path.join(path_to_output, output_fname)
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
