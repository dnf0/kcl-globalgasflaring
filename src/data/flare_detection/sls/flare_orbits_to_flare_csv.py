#!/apps/jasmin/jaspy/miniconda_envs/jaspy3.7/m3-4.6.14/envs/jaspy3.7-m3-4.6.14-r20200606/bin/python3

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
    proc_list = []

    cols = ['frp', 'swir_radiances', 'swir_radiances_22',
             'pixel_size', 'lats', 'lons', 'lats_arcmin',
            'lons_arcmin', 'year', 'month', 'day', 'hhmm', 'sensor',
            'frp_coeff', 'cloud_cover', 
            'sza', 'vza']
    dtypes = {'frp': float, 'swir_radiances': float, 
             'pixel_size': float, 'lats': float, 'lons': float, 'lats_arcmin': int,
            'lons_arcmin': int, 'year': int, 'month': int, 'day': int, 'hhmm': int, 'sensor': str,
            'se_dist': float, 'frp_coeff': float, 'cloud_cover': float,
             'sza': float, 'vza': float}

    csv_filepaths = glob.glob(fp.path_to_cems_output_l2 + '*/*/*/*/*_flares.csv')
    for f in csv_filepaths:
        if 'S3A' not in f:
            continue

        fname = f.split('/')[-1]
        if fname[0:40] in proc_list:  # don't process duplicate data
            print(fname)
            continue
        else:
            proc_list.append(fname[0:40])
        try:
            df_list.append(pd.read_csv(f, usecols=cols, dtype=dtypes))
        except Exception as e:
            print('Could not load df with error', e)
            continue
    df = pd.concat(df_list, ignore_index=True)

    path_to_out = os.path.join(fp.output_l3, 'all_sensors')
    if not os.path.exists(path_to_out):
        os.makedirs(path_to_out)
    df.to_csv(os.path.join(path_to_out, 'all_flares_sls_adaptive.csv'))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
