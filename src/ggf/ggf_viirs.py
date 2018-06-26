'''
Script to process VIIRS data to extract flaring
'''


import logging
import os

import h5py
import numpy as np
import pandas as pd
import scipy.constants as constants
import matplotlib.pyplot as plt


def compute_viirs_pixel_size(viirs_shape):
    samples = np.arange(viirs_shape[1])
    scan_angles = [compute_scan_angle(s) for s in samples]

    # from Elvidge et al 2013
    along_scan_size = [compute_along_scan_size(s) for s in samples]
    along_track_size = [compute_along_track_size(s) for s in samples]
    pixel_size = [s * t for s, t in zip(along_scan_size, along_track_size)]

    # now make same shape as input array
    pixel_size = np.array(pixel_size)
    pixel_size = np.tile(pixel_size, (viirs_shape[0], 1))

    scan_angles = np.rad2deg(np.array(scan_angles))
    scan_angles = np.tile(scan_angles, (viirs_shape[0], 1))

    return pixel_size, scan_angles


def compute_scan_angle(sample):
    s_scan = 0.000312  # IFOV, units in radians (from VIIRS_SDR_Users_guide.pdf fig 2)
    if sample >= 0 and sample < 640:
        n_pixels = sample
    elif sample >= 640 and sample < (640 + 368):
        n_pixels = 640 + (sample - 640) * 2
    elif sample >= (640 + 368) and sample < (640 + 368 + 592 * 2):
        n_pixels = 640 + 368 * 2 + (sample - (640 + 368)) * 3
    elif sample >= (640 + 368 + 592 * 2) and sample < (640 + 368 * 2 + 592 * 2):
        n_pixels = 640 + 368 * 2 + 592 * 2 * 3 + (sample - (640 + 368 + 593 * 2)) * 2
    else:
        n_pixels = 640 + 368 * 2 * 2 + 592 * 2 * 3 + (sample - (640 + 368 * 2 + 592 * 2))

    total_pixels = (592 * 3 + 368 * 2 + 640) * 2
    scan_angle = -0.5 * total_pixels * s_scan + 0.5 * s_scan + (n_pixels - 1) * s_scan
    return scan_angle


def compute_along_scan_size(sample):
    px_s = 0.776
    earth_rad = 6378.137
    h = 833

    ifov = px_s / h  # IFoV
    r = earth_rad + h

    scan_angle = compute_scan_angle(sample)

    size_scan = earth_rad * ifov * \
                (np.cos(scan_angle)
                 / (((earth_rad / r) ** 2 - (np.sin(scan_angle)) ** 2) ** 0.5) - 1)
    if np.abs(np.rad2deg(scan_angle)) <= 31.72:
        size_scan *= 1
    elif np.abs(np.rad2deg(scan_angle)) > 44.86:
        size_scan *= (1 / 3.)
    else:
        size_scan *= (1 / 1.5)
    return size_scan


def compute_along_track_size(sample):
    px_t = 0.742
    earth_rad = 6378.137
    h = 833

    ifov = px_t / h
    r = earth_rad + h

    scan_angle = compute_scan_angle(sample)

    return r * ifov * (np.cos(scan_angle) - \
                       ((earth_rad / r) ** 2 - (np.sin(scan_angle) ** 2)) ** 0.5)

def compute_swir_power(df):

    pixel_size = compute_along_scan_size(df['Sample_M10'].values) * \
                     compute_along_track_size(df['Sample_M10'].values) * 1000000  # convert to M**2
    frp = ((pixel_size * constants.sigma / 7.746e-09)
            * df['Rad_M10'].values) / 1000000
    return frp  # in MW



def get_sdrs(p):
    file_list = []
    for f in os.listdir(p):
        if 'SVM' in f:
            file_list.append(f)
    return file_list


def get_geo(p, sdr_file):
    for f in os.listdir(p):
        if (f[10:30] == sdr_file[10:30]) & ('SVM' not in f):
            return f
    return None


def read_sdr(p):
    ds = h5py.File(p)
    rad = ds['All_Data']['VIIRS-M10-SDR_All']['Radiance'][:]

    rad_factors = ds['All_Data']['VIIRS-M10-SDR_All']['RadianceFactors'][:]
    slope = rad_factors[0]
    offset = rad_factors[1]

    mask = rad != 65533
    rad = rad * slope + offset

    return rad, mask


def read_geo(p):
    ds = h5py.File(p)
    try:
        lats = ds["All_Data"]['VIIRS-MOD-GEO-TC_All']['Latitude'][:]
        lons = ds["All_Data"]['VIIRS-MOD-GEO-TC_All']['Longitude'][:]
        sza = ds["All_Data"]['VIIRS-MOD-GEO-TC_All']['SatelliteZenithAngle'][:]
    except:
        lats = ds["All_Data"]['VIIRS-MOD-GEO_All']['Latitude'][:]
        lons = ds["All_Data"]['VIIRS-MOD-GEO_All']['Longitude'][:]
        sza = ds["All_Data"]['VIIRS-MOD-GEO_All']['SatelliteZenithAngle'][:]

    return lats, lons, sza


def detect_hotspots(rad, mask):
    n_std = 4
    masked_rad = rad[mask]
    hotspot_mask = rad > np.mean(masked_rad) + n_std * np.std(masked_rad)
    return hotspot_mask & mask


def characterise_hotspots():
    pass


def main():

    # get viirs files
    #path_to_data = '/Volumes/INTENSO/globalgasflaring/viirs'
    #path_to_output = '/Volumes/INTENSO/globalgasflaring/viirs_hotspot_iraq.csv'
    #viirs_sdrs = get_sdrs(path_to_data)


    path_to_data = '/Users/danielfisher/Desktop/viirs_dl/libya'
    path_to_output = '/Users/danielfisher/Desktop/viirs_dl/libya/libya.csv'
    viirs_sdrs = get_sdrs(path_to_data)

    # set up output dataframe
    df_list = []

    # iterate over viirs files
    for sdr_file in viirs_sdrs:

        # get get file
        geo_file = get_geo(path_to_data, sdr_file)
        if geo_file is None:
            continue

        # get data
        try:
            rad, rad_mask = read_sdr(os.path.join(path_to_data, sdr_file))
            lats, lons, sza = read_geo(os.path.join(path_to_data, geo_file))
            areas, scan_angles = compute_viirs_pixel_size(rad.shape)
        except Exception, e:
            print 'Could not process', sdr_file, 'with error:', e
            continue

        if np.min(lats) < -90:
            print 'Could not process', sdr_file, 'due to files errors'
            continue

        # now build the masks
        sza_mask = sza < 22
        valid_mask = rad_mask & sza_mask

        # extract hotspots
        hotspots = detect_hotspots(rad, valid_mask)

        # make the dataframe
        df = pd.DataFrame()
        df['swir'] = rad[hotspots]
        df['area'] = areas[hotspots] * 1000000
        df['frp'] = ((df['area'] * 7.3199070488) * df['swir']) / 1000000
        df['lat'] = lats[hotspots]
        df['lon'] = lons[hotspots]
        df['year'] = sdr_file[11:15]
        df['month'] = sdr_file[15:17]
        df['day'] = sdr_file[17:19]
        df['hhmm'] = sdr_file[21:25]
        df['sensor'] = 'viirs'

        df_list.append(df)

    # concat df
    final_df = pd.concat(df_list)

    # write out
    final_df.to_csv(path_to_output)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
