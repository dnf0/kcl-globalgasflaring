#!/home/users/dnfisher/soft/virtual_envs/ggf/bin/python2

import os
import sys
import logging
from datetime import datetime

import epr
import numpy as np
from netCDF4 import Dataset
import pandas as pd


import src.config.constants as proc_const
import src.config.filepaths as fp


def read_atsr(path_to_ats_data):
    return epr.Product(path_to_ats_data)



def make_night_mask(ats_product):
    solar_elev_angle = np.deg2rad(ats_product.get_band('sun_elev_nadir').read_as_array())
    solar_zenith_angle = np.rad2deg(np.arccos(np.sin(solar_elev_angle)))
    return solar_zenith_angle >= proc_const.day_night_angle


def make_cloud_mask():
    pass


def get_swir_ref():
    pass


def find_closest_pixel():
    pass


def main():

    # read in the atsr prodcut and land water
    path_to_data = sys.argv[1]
    path_to_output = sys.argv[2]
    atsr_data = read_atsr(path_to_data)

    # set up ats data
    night_mask = make_night_mask(atsr_data)
    cloud_mask = make_cloud_mask()
    swir_ref = get_swir_ref()

    # lets get atsr orbit time
    orbit_time = datetime(1996, 6, 1)

    # load in the flare dataframe
    flare_df = pd.read_csv(fp.path_to_flare_df)

    # now subset down the dataframe by time to only those flares
    # that have been seen burning during this orbit
    flare_df = flare_df[(flare_df.dt_start <= orbit_time) &
                        (flare_df.dt_stop >= orbit_time)]

    # set up lists to hold flare id's and the orbit
    flare_id = []
    flare_orbit = []

    # now iterate over subset of flares and check if seen in this orbit
    for flare in flare_df:

        # get the closest pixel line and sample for the flare in this orbit
        x, y = find_closest_pixel(flare)

        # check if pixel is nighttime in this orbit if not continue
        if not night_mask[y, x]:
            continue

        # check if pixel is flare or is not cloudy
        is_flare = swir_ref[y, x]
        cloudy = cloud_mask[y, x]
        if is_flare and not cloudy:

            # if flare or not cloudy, record sampling
            flare_id.append(flare.index)
            flare_orbit.append(fname)


    # write out the recorded flare id's for this orbit
    output_fname = atsr_data.id_string.split('.')[0] + '_sampling.csv'
    if proc_const.sensor.upper() not in output_fname:
        output_fname = output_fname.replace(output_fname[0:3], proc_const.sensor.upper())
    csv_path = os.path.join(path_to_output, output_fname)
    df.to_csv(csv_path)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()