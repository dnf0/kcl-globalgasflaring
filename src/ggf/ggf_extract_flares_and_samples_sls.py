#!/apps/contrib/jaspy/miniconda_envs/jaspy2.7/m2-4.6.14/envs/jaspy2.7-m2-4.6.14-r20190715/bin/python2.7

import os
import sys
import logging

import numpy as np
import pandas as pd
import scipy.ndimage as ndimage

import src.config.constants as proc_const
import src.config.filepaths as fp
import src.ggf.ggf_extract_hotspots_sls as ggf_extract_hotspots_sls
import src.ggf.ggf_extract_flares_and_samples_atx as ggf_extract_flares_and_samples_atx

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def merge_hotspot_dataframes(ats_hotspot_df, sls_hotspot_df):
    ats_hotspot_df['sensor'] = 1
    sls_hotspot_df['sensor'] = -1
    appended_df = ats_hotspot_df.append(sls_hotspot_df)
    hotspot_df = appended_df.groupby(['lats_arcmin', 'lons_arcmin'], as_index=False).agg({'sensor': np.sum})

    # the below information is not used here but we can use this
    # idea to update the sensor information during the post processing
    hotspot_df.sensor.loc[hotspot_df.sensor == 1] = 'atx'
    hotspot_df.sensor.loc[hotspot_df.sensor == -1] = 'sls'
    hotspot_df.sensor.loc[hotspot_df.sensor == 0] = 'both'
    return hotspot_df


def get_arcmin_int(x):
    '''
    rounds the data decimal fraction of a degree
    to the nearest arc minute
    '''
    neg_values = x < 0

    abs_x = np.abs(x)
    floor_x = np.floor(abs_x)
    decile = abs_x - floor_x
    minute = np.around(decile * 60)  # round to nearest arcmin
    minute_fraction = minute * 0.01  # convert to fractional value (ranges from 0 to 0.6)

    max_minute = minute_fraction > 0.59

    floor_x[neg_values] *= -1
    floor_x[neg_values] -= minute_fraction[neg_values]
    floor_x[~neg_values] += minute_fraction[~neg_values]

    # deal with edge cases, and just round them all up
    if np.sum(max_minute) > 0:
        floor_x[max_minute] = np.around(floor_x[max_minute])

    # now to get rid of rounding errors and allow comparison multiply by 100 and convert to int
    floor_x = (floor_x * 100).astype(int)

    return floor_x


def round_to_arcmin(x, dec=20, base=60. / 3600):
    return np.round(base * np.round(x / base), dec)


def compute_pixel_size(samples):
    pix_sizes = np.loadtxt(fp.path_to_sls_pix_sizes)
    pix_sizes *= 1000000  # convert from km^2 to m^2
    return pix_sizes[samples]


def construct_hotspot_line_sample_df(s3_data, hotspot_mask):
    lines, samples = np.where(hotspot_mask)
    lats = s3_data['geodetic_an']['latitude_an'][:][hotspot_mask]
    lons = s3_data['geodetic_an']['longitude_an'][:][hotspot_mask]

    # round geographic data to desired reoslution
    lats_arcmin = get_arcmin_int(lats)
    lons_arcmin = get_arcmin_int(lons)

    df = pd.DataFrame()
    datasets = [lats_arcmin, lons_arcmin, lines, samples]
    names = ['lats_arcmin', 'lons_arcmin', 'lines', 'samples']
    for k, v in zip(names, datasets):
        df[k] = v
    return df


def construct_hotspot_df(flare_df, cloud_cover,
                         s3_data, resolution, sza, vza, sensor):
    coords = [flare_df.lines.values, flare_df.samples.values]

    lats = s3_data['geodetic_an']['latitude_an'][:][coords]
    lons = s3_data['geodetic_an']['longitude_an'][:][coords]

    # round geographic data to desired reoslution
    rounded_lats = round_to_arcmin(lats, base=resolution)
    rounded_lons = round_to_arcmin(lons, base=resolution)

    swir_radiances = s3_data['S5_radiance_an']['S5_radiance_an'][:].filled(0)[coords]
    swir_reflectances = swir_radiances / 254.23103333 * np.pi * 100

    swir_radiances_22 = s3_data['S6_radiance_an']['S6_radiance_an'][:].filled(0)[coords]

    pixel_size = compute_pixel_size(flare_df.samples.values)
    frp = ggf_extract_flares_and_samples_atx.compute_frp(swir_radiances, pixel_size, sensor)

    sza_subset = sza[coords]
    vza_subset = vza[coords]

    cc = cloud_cover[coords]

    datasets = [rounded_lats, rounded_lons, frp, cc,
                swir_radiances, swir_reflectances, swir_radiances_22,
                sza_subset, vza_subset, pixel_size]
    names = ['lats', 'lons', 'frp', 'cloud_cover',
             'swir_radiances', 'swir_reflectances', 'swir_radiances_22',
             'sza', 'vza', 'pixel_size']
    for k, v in zip(names, datasets):
        flare_df[k] = v

    return flare_df


def construct_sample_df(flare_df, s3_data, cloud_cover, night_mask):

    # get geographic coordinates
    lats = s3_data['geodetic_an']['latitude_an'][:]
    lons = s3_data['geodetic_an']['longitude_an'][:]
    lats_arcmin = get_arcmin_int(lats)
    lons_arcmin = get_arcmin_int(lons)

    # set up df for merging
    orbit_df = pd.DataFrame()
    orbit_df['lats_arcmin'] = lats_arcmin[night_mask]
    orbit_df['lons_arcmin'] = lons_arcmin[night_mask]
    orbit_df['cloud_cover'] = cloud_cover[night_mask]

    orbit_df = orbit_df.merge(flare_df, on=['lats_arcmin', 'lons_arcmin'])

    return orbit_df


def group_hotspot_df(df):
    agg_dict = {'frp': np.sum,  # sum to get the total FRP in the grid cell
                'swir_radiances': np.mean,
                'swir_radiances_22': np.mean,
                'pixel_size': np.sum,
                'lats': np.mean,
                'lons': np.mean,
                'cloud_cover': np.mean,
                'sza': np.mean,
                'vza': np.mean,
                }
    return df.groupby(['lats_arcmin', 'lons_arcmin'], as_index=False).agg(agg_dict)


def group_sample_df(df):
    return df.groupby(['lats_arcmin', 'lons_arcmin'], as_index=False).agg({'cloud_cover': np.mean})


def extend_df(df, ymdhm, hotspot_df=False):
    df['year'] = int(ymdhm[0:4])
    df['month'] = int(ymdhm[4:6])
    df['day'] = int(ymdhm[6:8])
    df['hhmm'] = int(ymdhm[9:])
    df['sensor'] = 'sls'

    if hotspot_df:
        df['frp_coeff'] = proc_const.frp_coeff['sls']


def main():
    # some processing constants
    resolution = 60 / 3600.  # TODO MOVE TO CONSTANTS

    # load in the persistent flare location dataframe
    ats_flare_df = pd.read_csv(os.path.join(fp.path_to_cems_output_l3, 'all_sensors', 'all_flare_locations_ats.csv'))
    sls_flare_df = pd.read_csv(os.path.join(fp.path_to_cems_output_l3, 'all_sensors', 'all_flare_locations_sls.csv'))

    # merge dataframe
    flare_df = merge_hotspot_dataframes(ats_flare_df, sls_flare_df)

    # read in the atsr product
    path_to_data = sys.argv[1]
    path_to_output = sys.argv[2]
    path_to_temp = sys.argv[3]

    try:
        # get ymd
        ymdhm = path_to_data.split('/')[-1][16:29]

        s3_data = ggf_extract_hotspots_sls.extract_zip(path_to_data, path_to_temp)

        sza, night_mask = ggf_extract_hotspots_sls.make_night_mask(s3_data)
        if night_mask.max() == 0:
            return

        vza, vza_mask = ggf_extract_hotspots_sls.make_vza_mask(s3_data)

        potential_hotspot_mask = ggf_extract_hotspots_sls.detect_hotspots(s3_data)
        is_not_cloud_mask = ggf_extract_hotspots_sls.make_cloud_mask(s3_data)

        valid_mask = night_mask & vza_mask

        hotspot_mask = valid_mask & potential_hotspot_mask
        cloud_mask = valid_mask & ~potential_hotspot_mask & ~is_not_cloud_mask

        # get cloud cover map from cloud mask
        bg_size = 2*16 + 1  #  16 as 500m resolution TODO MOVE TO CONFIG
        k = np.ones([bg_size, bg_size])
        s = ndimage.convolve(cloud_mask.astype(int), k, mode='constant', cval=0.0)
        count = ndimage.convolve(np.ones(cloud_mask.shape), k, mode='constant', cval=0.0)
        cloud_cover = s/count

        # do the processing for samples, where we just get the cloud cover for each location
        sample_df = construct_sample_df(flare_df, s3_data, cloud_cover, valid_mask)
        grouped_sample_df = group_sample_df(sample_df)
        extend_df(grouped_sample_df, ymdhm)

        sample_csv_path = path_to_output.replace('hotspots.csv', 'sampling.csv')
        logger.info(sample_csv_path)
        grouped_sample_df.to_csv(sample_csv_path, index=False)

        # do the processing for persistent hotspots
        if np.sum(hotspot_mask):
            hotspot_line_sample_df = construct_hotspot_line_sample_df(s3_data, hotspot_mask)
            persistent_hotspot_line_sample_df = pd.merge(flare_df, hotspot_line_sample_df, on=['lats_arcmin', 'lons_arcmin'])
            persistent_hotspot_df = construct_hotspot_df(persistent_hotspot_line_sample_df,
                                                         cloud_cover,
                                                         s3_data,
                                                         resolution,
                                                         sza,
                                                         vza,
                                                         'sls')
            grouped_persistent_hotspot_df = group_hotspot_df(persistent_hotspot_df)
            extend_df(grouped_persistent_hotspot_df, ymdhm, hotspot_df=True)

            flare_csv_path = path_to_output.replace('hotspots.csv', 'flares.csv')
            logger.info(flare_csv_path)
            grouped_persistent_hotspot_df.to_csv(flare_csv_path, index=False)

    except Exception as e:
        logger.warning('Flare and sample generation failed with error: ' + str(e))
        # dump the csvs even if we fail
        with open(path_to_output.replace('hotspots.csv', 'flares.csv'), "w"):
            pass
        with open(path_to_output.replace('hotspots.csv', 'sampling.csv'), "w"):
            pass

if __name__ == "__main__":
    main()
