#!/home/users/dnfisher/soft/virtual_envs/ggf/bin/python2

'''
Take all the flare.csv files for the ATSR
sensors and put them into a single csv file
for the entire time series
'''

import logging
import glob
import os

import pandas as pd

import src.config.filepaths as fp

def main():

    df_list = []
    cols = ['frp', 'swir_radiances', 'mwir_radiances', 'mwir_bg',
             'pixel_size', 'lats', 'lons', 'lats_arcmin',
            'lons_arcmin', 'year', 'month', 'day', 'hhmm', 'sensor',
            'se_dist', 'frp_coeff', 'cloud_cover', 'hotspot_bg_pc', 'inval_pixels_bg_pc', 'bg_size_used']
    dtypes = {'frp': float, 'swir_radiances': float, 'mwir_radiances': float, 'mwir_bg': float,
             'pixel_size': float, 'lats': float, 'lons': float, 'lats_arcmin': int,
            'lons_arcmin': int, 'year': int, 'month': int, 'day': int, 'hhmm': int, 'sensor': str,
            'se_dist': float, 'frp_coeff': float, 'cloud_cover': float, 'hotspot_bg_pc': float,
            'inval_pixels_bg_pc': float, 'bg_size_used': int}

    csv_filepaths = glob.glob(fp.path_to_cems_output_l2 + '*/*/*/*/*_flares.csv')
    for f in csv_filepaths:
        if 'S3A' in f:
            continue
        print f
        df_list.append(pd.read_csv(f, usecols=cols, dtype=dtypes))

    df = pd.concat(df_list, ignore_index=True)

    path_to_out = os.path.join(fp.path_to_cems_output_l3, 'all_sensors')
    if not os.path.exists(path_to_out):
        os.makedirs(path_to_out)
    df.to_csv(os.path.join(path_to_out, 'all_flares_atx_adaptive.csv'))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
