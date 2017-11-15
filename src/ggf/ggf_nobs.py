#!/home/users/dnfisher/soft/virtual_envs/ggf/bin/python2

import os
import sys
import logging
from datetime import datetime

import epr
import numpy as np
import scipy.spatial as spatial
import pandas as pd

import matplotlib.pyplot as plt


import src.config.constants as proc_const
import src.config.filepaths as fp


def read_atsr(path_to_ats_data):
    return epr.Product(path_to_ats_data)


def make_night_mask(ats_product):
    solar_elev_angle = np.deg2rad(ats_product.get_band('sun_elev_nadir').read_as_array())
    solar_zenith_angle = np.rad2deg(np.arccos(np.sin(solar_elev_angle)))
    return solar_zenith_angle >= proc_const.day_night_angle


def make_cloud_mask(ats_product):
    cloud_mask = ats_product.get_band('cloud_flags_nadir').read_as_array()
    return cloud_mask <= 1  # masking needs to be checked


def get_swir_ref(ats_product):
    return ats_product.get_band('reflec_nadir_1600').read_as_array()


def myround(x, dec=20, base=60. / 3600):
    return np.round(base * np.round(x/base), dec)


def get_geo_data(ats_product):
    lats = myround(ats_product.get_band('latitude').read_as_array())
    lons = myround(ats_product.get_band('longitude').read_as_array())
    return lats, lons


def find_closest_pixel():
    pass


def main():

    # some processing constants
    resolution = 60 / 3600.  # Degrees. same as with monthly aggregation

    # read in the atsr prodcut
    # path_to_data = sys.argv[1]
    # path_to_output = sys.argv[2]
    path_to_data = '/Users/danielfisher/Projects/kcl-globalgasflaring/data/raw/atsr/ATS_TOA_1PUUPA20090608_002920_000065272079_00388_38017_1539.N1'
    path_to_output = None
    atsr_data = read_atsr(path_to_data)

    # set up ats data
    night_mask = make_night_mask(atsr_data)
    cloud_free_mask = make_cloud_mask(atsr_data)
    swir_ref = get_swir_ref(atsr_data)
    rounded_lats, rounded_lons = get_geo_data(atsr_data)

    valid_round_lats = rounded_lats[night_mask & cloud_free_mask ]
    valid_round_lons = rounded_lons[night_mask & cloud_free_mask ]

    # set up the cKDTree for querying flare locations
    combined_lat_lon = np.dstack([valid_round_lats.ravel(), valid_round_lons.ravel()])[0]
    orbit_kdtree = spatial.cKDTree(combined_lat_lon)

    # get atsr orbit time
    year = int(atsr_data.id_string[14:18])
    month = int(atsr_data.id_string[18:20])
    day = int(atsr_data.id_string[20:22])
    orbit_time = datetime(year, month, day)

    # load in the flare dataframe
    root = fp.path_to_test_csv_out
    flare_df = pd.read_csv(os.path.join(root, 'all_flares.csv'))

    #flare_df.reset_index(inplace=True)  # get the flare ids

    flare_df['dt_start'] = pd.to_datetime(flare_df['dt_start'])
    flare_df['dt_stop'] = pd.to_datetime(flare_df['dt_stop'])

    # now subset down the dataframe by time to only those flares
    # that have been seen burning during this orbit
    flare_df = flare_df[(flare_df.dt_start <= orbit_time) &
                        (flare_df.dt_stop >= orbit_time)]
    if flare_df.empty:
        return

    # set up the flare lats and lons for assessment in kdtree
    flare_lat_lon = np.array(zip(flare_df.lats.values, flare_df.lons.values))

    # get the indexes inside the orbit
    distances, indexes = orbit_kdtree.query(flare_lat_lon)
    valid_distances = distances < resolution
    flare_ids = flare_df.index[valid_distances]

    output_df = pd.DataFrame({'flare_ids': flare_ids})

    # write out the recorded flare id's for this orbit
    output_fname = atsr_data.id_string.split('.')[0] + '_sampling.csv'
    if proc_const.sensor.upper() not in output_fname:
        output_fname = output_fname.replace(output_fname[0:3], proc_const.sensor.upper())
    csv_path = os.path.join(path_to_output, output_fname)
    output_df.to_csv(csv_path)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()