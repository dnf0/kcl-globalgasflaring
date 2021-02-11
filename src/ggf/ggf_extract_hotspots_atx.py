#!/apps/jasmin/jaspy/miniconda_envs/jaspy3.7/m3-4.6.14/envs/jaspy3.7-m3-4.6.14-r20200606/bin/python3

'''
Detects hotspot locations in a given file.  Used later
to detect flares based on hotspot persistency.
'''

import sys
import logging

import epr
import numpy as np
import pandas as pd

import src.config.constants as proc_const

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

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


def detect_hotspots(ats_product):
    swir = ats_product.get_band('reflec_nadir_1600').read_as_array()
    nan_mask = np.isnan(swir)  # get rid of SWIR nans also
    return (swir > proc_const.swir_thresh_ats) & ~nan_mask


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


def run(path_to_data, path_to_output):

    atsr_data = read_atsr(path_to_data)

    # get day/night mask first
    night_mask = make_night_mask(atsr_data)
    logger.info('Night mask samples ' + str(np.sum(night_mask)))
    logger.info('Night mask shape ' + str(night_mask.shape))

    # get nighttime flare mask
    potential_hotspot_mask = detect_hotspots(atsr_data)
    logger.info('potential_hotspot_mask samples ' + str(np.sum(potential_hotspot_mask)))
    logger.info('potential_hotspot_mask shape ' + str(potential_hotspot_mask.shape))
    hotspot_mask = potential_hotspot_mask & night_mask
    logger.info('hotspot_mask samples ' + str(np.sum(hotspot_mask)))
    logger.info('hotspot_mask shape ' + str(hotspot_mask.shape))

    # if we exceed this number of flares, then likely
    # something wrong with orbit and reject
    max_flares_in_an_orbit = 20000  # todo add to constants
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


def main():
    # read in the atsr product
    path_to_data = sys.argv[1]
    path_to_output = sys.argv[2]
    run(path_to_data, path_to_output)


if __name__ == "__main__":
    main()
