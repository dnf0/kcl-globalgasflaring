#!/home/users/dnfisher/soft/virtual_envs/ggf/bin/python2

import os
import sys
import logging
import zipfile
import shutil

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from scipy.interpolate import RectBivariateSpline

import src.config.constants as proc_const
import src.config.filepaths as fp


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


def detect_hotspots_non_parametric(ds, sza_mask, vza_mask):

    # first get unillimunated central swath data
    valid_mask = ds != -999
    useable_data = ds[sza_mask & vza_mask & valid_mask]

    # find threshold for data
    useable_data.sort()
    top_subset = useable_data[-1000:]
    bottom_subset = useable_data[:1000]

    diff_top = top_subset[1:] - top_subset[0:-1]

    # now get the smallest non-zero difference for the top 1k values
    not_zero = diff_top != 0
    smallest_diff = np.min(diff_top[not_zero])

    logger.info('smallest diff for top 1k : ' + str(smallest_diff))

    diff_mask = diff_top > smallest_diff
    thresh = np.min(top_subset[1:][diff_mask])

    logger.info('Threshold using scene smallest diff: ' + str(thresh))

    # get hotspots
    above_thresh = ds > thresh

    return valid_mask & above_thresh


def make_cloud_mask(s3_data):
    # over land or water and cloud free (i.e. bit 0 is set (cloud free land)  or unset(cloud free water))
    return s3_data['flags_an']['cloud_an'][:] == 0


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


def myround(x, dec=20, base=60. / 3600):
    return np.round(base * np.round(x / base), dec)


def compute_pixel_size(samples):
    pix_sizes = np.loadtxt(fp.path_to_sls_pix_sizes)
    pix_sizes *= 1000000  # convert from km^2 to m^2
    return pix_sizes[samples]


def compute_frp(pixel_radiances, pixel_sizes, sensor):
    return pixel_sizes * proc_const.frp_coeff[sensor] * pixel_radiances / 1000000  # in MW


def construct_hotspot_line_sample_df(s3_data, hotspot_mask):
    lines, samples = np.where(hotspot_mask)
    lats = s3_data['geodetic_an']['latitude_an'][:][hotspot_mask]
    lons = s3_data['geodetic_an']['longitude_an'][:][hotspot_mask]

    # round geographic data to desired reoslution
    lats_arcmin = get_arcmin(lats)
    lons_arcmin = get_arcmin(lons)

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
    rounded_lats = myround(lats, base=resolution)
    rounded_lons = myround(lons, base=resolution)

    swir_radiances = s3_data['S5_radiance_an']['S5_radiance_an'][:].filled(0)[coords]
    swir_reflectances = swir_radiances / 254.23103333 * np.pi * 100

    swir_radiances_22 = s3_data['S6_radiance_an']['S6_radiance_an'][:].filled(0)[coords]

    pixel_size = compute_pixel_size(flare_df.samples.values)
    frp = compute_frp(swir_radiances, pixel_size, sensor)

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
    lats_arcmin = get_arcmin(lats)
    lons_arcmin = get_arcmin(lons)

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

    #data = 'S3A_SL_1_RBT____20180104T185242_20180104T185542_20180105T224332_0179_026_213_6419_LN2_O_NT_002.zip'
    #path_to_data = os.path.join('/neodc/sentinel3a/data/SLSTR/L1_RBT/2018/01/04', data)
    #path_to_output = fp.path_to_temp
    #path_to_temp = fp.path_to_temp

    # get ymd
    ymdhm = path_to_data.split('/')[-1][16:29]

    s3_data = extract_zip(path_to_data, path_to_temp)

    # load in S5 and S6 channels
    s5_data = s3_data['S5_radiance_an']['S5_radiance_an'][:].filled(-999)

    sza, night_mask = make_night_mask(s3_data)
    if night_mask.max() == 0:
        return

    vza, vza_mask = make_vza_mask(s3_data)

    potential_hotspot_mask = detect_hotspots_non_parametric(s5_data, night_mask, vza_mask)
    is_not_cloud_mask = make_cloud_mask(s3_data)

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
    sample_output_fname = path_to_data.split('/')[-1].split('.')[0] + '_sampling.csv'
    sample_csv_path = os.path.join(path_to_output, sample_output_fname)
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
        flare_output_fname = path_to_data.split('/')[-1].split('.')[0] + '_flares.csv'
        flare_csv_path = os.path.join(path_to_output, flare_output_fname)
        grouped_persistent_hotspot_df.to_csv(flare_csv_path, index=False)
    return

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
