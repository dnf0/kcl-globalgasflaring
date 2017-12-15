#!/home/users/dnfisher/soft/virtual_envs/ggf/bin/python2

"""
code description:

For a given ATSR orbit we determine if any of the gas flares in our detected
gas flare file have a potential observation.  This is done as:

1. Get the start time of the ATSR file
2. Restrict only those conditions where the flare might have been observed.  That is
it being a night-time scene and either a) a flaring location (i.e. ref > 0.1) or b) a
cloud free location (i.e. free from any cloud).  This gives us all possibly
flare observation opportunities.
3. Extract the lats and lons for these potential flaring sites in the ATSR orbit.
4. Resample the ATSR lats and Lons to 1 arc minute resolution (same as monhtly aggregation)

5. Form a KDtree using these ATSR lats and lons
6. From the gas flare location DF. Get only those flares that have been seen
burning before and after the ATSR overpass. This means that they were operating
at the time of the ATSR overpass.
7. Determine whether these flaring locations are contained within the ATSR orbit
through comparison against the KDTree.
8. The KDtree returns distances, any flare that is located in the atsr grid at a
distance less the resampling resolution is considered to be observed in this orbit
9. For these flares write out the flare ID, along with the ATSR lat and lon that
was matched to the flare.  The lats and lons are used to make sure that the flares
that we see are actually reasonable in terms of thier locations. I.e. as a sanity check.
"""

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
    # over land or water and cloud free (i.e. bit 0 is set (cloud free land)  or unset(cloud free water))
    return cloud_mask <= 1


def get_swir_mask(ats_product):
    swir_ref = ats_product.get_band('reflec_nadir_1600').read_as_array()
    return swir_ref > proc_const.swir_thresh


def myround(x, dec=20, base=60. / 3600):
    return np.round(base * np.round(x/base), dec)


def get_geo_data(ats_product, mask):

    # first mask lats and lons
    lats = ats_product.get_band('latitude').read_as_array()[mask]
    lons = ats_product.get_band('longitude').read_as_array()[mask]

    # then round them
    rounded_lats = myround(lats)
    rounded_lons = myround(lons)

    # then get the unique lat and lon combinations
    unique_coords = set(zip(rounded_lats, rounded_lons))

    # return them unzipped
    rounded_lats, rounded_lons = zip(*unique_coords)
    return rounded_lats, rounded_lons


def main():

    # some processing constants
    resolution = 60 / 3600.  # Degrees. same as with monthly aggregation

    # read in the atsr prodcut
    path_to_data = sys.argv[1]
    path_to_output = sys.argv[2]
    atsr_data = read_atsr(path_to_data)

    # set up masks that define potential flaring sites
    night_mask = make_night_mask(atsr_data)
    cloud_free_mask = make_cloud_mask(atsr_data)
    swir_mask = get_swir_mask(atsr_data)

    # get the rounded lats and lons of the potential flaring sites
    potential_flares = cloud_free_mask | swir_mask  # either cloud free or high swir
    flare_mask = night_mask & potential_flares  # and also at night
    rounded_lats, rounded_lons = get_geo_data(atsr_data, flare_mask)

    # set up the cKDTree for querying flare locations
    combined_lat_lon = np.dstack([rounded_lats, rounded_lons])[0]
    orbit_kdtree = spatial.cKDTree(combined_lat_lon)

    # get atsr orbit time
    year = int(atsr_data.id_string[14:18])
    month = int(atsr_data.id_string[18:20])
    day = int(atsr_data.id_string[20:22])
    orbit_time = datetime(year, month, day)

    # load in the flare dataframe
    flare_df = pd.read_csv(os.path.join(fp.path_to_cems_output_l3, 'all_sensors', 'all_flares.csv'))
    flare_df['dt_start'] = pd.to_datetime(flare_df['dt_start'])
    flare_df['dt_stop'] = pd.to_datetime(flare_df['dt_stop'])
    flare_df['ids'] = flare_df.index
    logger.info('Max flare index: ' + str(flare_df.index.max()))

    # now subset down the dataframe by time to only those flares
    # that have been seen burning before AND after this orbit
    flare_df = flare_df[(flare_df.dt_start <= orbit_time) &
                        (flare_df.dt_stop >= orbit_time)]
    if flare_df.empty:
        return

    # set up the flare lats and lons for assessment in kdtree
    flare_lat_lon = np.array(zip(flare_df.lats.values, flare_df.lons.values))

    # compare the flare locations to the potential locations in the orbit
    distances, indexes = orbit_kdtree.query(flare_lat_lon)

    # find the flaring locations in the orbit by distance measure
    valid_distances = distances <= resolution
    flare_ids = flare_df.ids[valid_distances].values
    matched_lats = combined_lat_lon[indexes[valid_distances], 0]
    matched_lons = combined_lat_lon[indexes[valid_distances], 1]

    # set up output df
    output_df = pd.DataFrame({'flare_ids': flare_ids,
                              'matched_lats': matched_lats,
                              'matched_lons': matched_lons
                              })

    # write out the recorded flare id's for this orbit
    output_fname = atsr_data.id_string.split('.')[0] + '_sampling.csv'
    if proc_const.sensor.upper() not in output_fname:
        output_fname = output_fname.replace(output_fname[0:3], proc_const.sensor.upper())
    csv_path = os.path.join(path_to_output, output_fname)
    output_df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
