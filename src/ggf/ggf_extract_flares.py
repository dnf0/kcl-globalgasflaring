#!/home/users/dnfisher/soft/virtual_envs/ggf/bin/python2

import os
import sys
import logging
from datetime import datetime

import epr
import numpy as np
import pandas as pd

import src.config.constants as proc_const
import src.models.atsr_pixel_size as atsr_pixel_size
import src.config.filepaths as fp


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
    return solar_zenith_angle >= proc_const.day_night_angle


def detect_hotspots(ats_product):
    swir = ats_product.get_band('reflec_nadir_1600').read_as_array()
    nan_mask = np.isnan(swir)  # get rid of SWIR nans also
    return (swir > proc_const.swir_thresh) & ~nan_mask


def make_cloud_mask(ats_product):
    cloud_mask = ats_product.get_band('cloud_flags_nadir').read_as_array()
    # over land or water and cloud free (i.e. bit 0 is set (cloud free land)  or unset(cloud free water))
    return cloud_mask <= 1


def get_arcmin(x):
    '''
    rounds the data decimal fraction of a degree
    to the nearest arc minute
    '''
    neg_values = x < 0

    abs_x = np.abs(x)
    floor_x = np.floor(abs_x)
    decile = abs_x - floor_x
    minute = np.around(decile * 60)  # round to nearest arcmin
    minute_fraction = minute*0.01  # convert to fractional value (ranges from 0 to 0.6)

    floor_x[neg_values] *= -1
    floor_x[neg_values] -= minute_fraction
    floor_x[~neg_values] += minute_fraction

    return floor_x


def myround(x, dec=20, base=60. / 3600):
    return np.round(base * np.round(x/base), dec)


def radiance_from_reflectance(pixels, ats_product, sensor):
    # convert from reflectance to radiance see Smith and Cox 2013
    return pixels / 100.0 * proc_const.solar_irradiance[sensor] * sun_earth_distance(ats_product) ** 2 / np.pi


def sun_earth_distance(id_string):
    doy = datetime.strptime(id_string[14:22], "%Y%m%d").timetuple().tm_yday
    return 1 + 0.01672 * np.sin(2 * np.pi * (doy - 93.5) / 365.0)


def compute_pixel_size(samples):
    pix_sizes = atsr_pixel_size.compute() * 1000000  # convert from km^2 to m^2
    return pix_sizes[samples]


def compute_frp(pixel_radiances, pixel_sizes, sensor):
    return pixel_sizes * proc_const.frp_coeff[sensor] * pixel_radiances / 1000000  # in MW


def construct_hotspot_df(product, resolution, hotspot_mask, sensor):

    lines, samples = np.where(hotspot_mask)
    lats = product.get_band('latitude').read_as_array()[hotspot_mask]
    lons = product.get_band('longitude').read_as_array()[hotspot_mask]

    # round geographic data to desired reoslution
    lats_arcmin = get_arcmin(lats)
    lons_arcmin = get_arcmin(lons)
    rounded_lats = myround(lats, base=resolution)
    rounded_lons = myround(lons, base=resolution)

    reflectances = product.get_band('reflec_nadir_1600').read_as_array()[hotspot_mask]
    solar_elev_angle = product.get_band('sun_elev_nadir').read_as_array()[hotspot_mask]
    view_elev_angle = product.get_band('view_elev_nadir').read_as_array()[hotspot_mask]

    radiances = radiance_from_reflectance(reflectances, product, sensor)
    pixel_size = compute_pixel_size(samples)
    frp = compute_frp(radiances, pixel_size, sensor)

    df = pd.DataFrame()
    datasets = [rounded_lats, rounded_lons, lats_arcmin, lons_arcmin, frp,
                radiances, reflectances, solar_elev_angle, view_elev_angle, pixel_size]
    names = ['lats', 'lons', 'lats_arcmin', 'lons_arcmin', 'frp',
             'radiances', 'reflectances', 'sun_elev', 'view_elev', 'pixel_size']
    for k, v in zip(names, datasets):
        df[k] = v
    return df


def construct_sample_df(product, resolution, cloud_free_mask, hotspot_mask, sample_mask):

    # assign type mask
    types = np.zeros(cloud_free_mask.shape)
    types[cloud_free_mask] = 2
    types[hotspot_mask] = 1
    types = types[sample_mask]

    lats = product.get_band('latitude').read_as_array()[sample_mask]
    lons = product.get_band('longitude').read_as_array()[sample_mask]

    # round geographic data to desired reoslution
    lats_arcmin = get_arcmin(lats)
    lons_arcmin = get_arcmin(lons)
    rounded_lats = myround(lats, base=resolution)
    rounded_lons = myround(lons, base=resolution)

    df = pd.DataFrame()
    datasets = [rounded_lats, rounded_lons, lats_arcmin, lons_arcmin, types]
    names = ['lats', 'lons', 'lats_arcmin', 'lons_arcmin', 'types']
    for k, v in zip(names, datasets):
        df[k] = v
    return df


def get_type(a):
    if np.in1d(1, a)[0]:  # if a gas flare pixel in grid cell, then return as gas flare type
        return 1
    else:
        return 2


def group_hotspot_df(df):
    return df.groupby(['lats_arcmin', 'lons_arcmin'], as_index=False).agg({'types': get_type,
                                                                           'lats': np.mean,
                                                                           'lons': np.mean})


def group_sample_df(df):
    agg_dict = {'frp': np.sum,  # sum to get the total FRP in the grid cell
                'radiances': np.sum,
                'reflectances': np.sum,
                'sun_elev': np.mean,
                'view_elev': np.mean,
                'pixel_size': np.sum,
                'lats': np.mean,
                'lons': np.mean}
    return df.groupby(['lats_arcmin', 'lons_arcmin'], as_index=False).agg(agg_dict)


def extend_df(df, sensor, id_string, hotspot_df=False):
    df['year'] = int(id_string[14:18])
    df['month'] = int(id_string[18:20])
    df['day'] = int(id_string[20:22])
    df['hhmm'] = int(id_string[23:27])
    df['sensor'] = sensor

    if hotspot_df:
        df['se_dist'] = sun_earth_distance(id_string)
        df['frp_coeff'] = proc_const.frp_coeff[sensor]


def main():

    # some processing constants
    resolution = 60 / 3600.  # Degrees. same as with monthly aggregation

    # load in the flare location dataframe
    flare_df = pd.read_csv(os.path.join(fp.path_to_cems_output_l3, 'all_sensors', 'all_flare_locations.csv'))

    # read in the atsr product
    path_to_data = sys.argv[1]
    path_to_output = sys.argv[2]
    atsr_data = read_atsr(path_to_data)
    sensor = define_sensor(path_to_data)

    # set up various data masks
    night_mask = make_night_mask(atsr_data)
    cloud_free_mask = make_cloud_mask(atsr_data)

    potential_hotspot_mask = detect_hotspots(atsr_data)
    potential_sample_mask = cloud_free_mask | potential_hotspot_mask  # cloud free or high SWIR

    hotspot_mask = night_mask & potential_hotspot_mask
    sample_mask = night_mask & potential_sample_mask

    # extract all relevant information for both sampling and flare dataframes
    hotspot_df = construct_hotspot_df(atsr_data, resolution, hotspot_mask, sensor)
    sample_df = construct_sample_df(atsr_data, resolution, cloud_free_mask, hotspot_mask, sample_mask)

    # group to rounded resolution (done in previous step) aggregating appropriately
    grouped_hotspot_df = group_hotspot_df(hotspot_df)
    grouped_sample_df = group_sample_df(sample_df)

    # extend grouped dataframes with relevant information
    extend_df(grouped_hotspot_df, sensor, atsr_data.id_string, hotspot_df=True)
    extend_df(grouped_sample_df, sensor, atsr_data.id_string)

    # extract data in orbit related to flares by merging
    flare_sample_df = pd.merge(flare_df, grouped_sample_df, on=['lats_arcmin', 'lons_arcmin'])
    flare_hotspot_df = pd.merge(flare_df, grouped_hotspot_df, on=['lats_arcmin', 'lons_arcmin'])

    # write out the recorded flare data for this orbit
    sample_output_fname = atsr_data.id_string.split('.')[0] + '_sampling.csv'
    flare_output_fname = atsr_data.id_string.split('.')[0] + '_flares.csv'

    sample_output_fname = sample_output_fname.replace(sample_output_fname[0:3], sensor.upper())
    flare_output_fname = flare_output_fname.replace(flare_output_fname[0:3], sensor.upper())

    sample_csv_path = os.path.join(path_to_output, sample_output_fname)
    flare_csv_path = os.path.join(path_to_output, flare_output_fname)

    flare_sample_df.to_csv(sample_csv_path, index=False)
    flare_hotspot_df.to_csv(flare_csv_path, index=False)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
