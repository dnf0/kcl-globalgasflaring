import os
import glob
import logging
from datetime import datetime

import pandas as pd
import numpy as np

import src.config.filepaths as fp


def main():

    df_list = []
    csv_files = glob.glob(os.path.join(fp.path_to_test_csv_out, "*", '*', "*_flaring_subset.csv"))

    for f in csv_files:
        df = pd.read_csv(f)

        split_f = f.split('/')
        year = split_f[-2]
        month = split_f[-1][0:2]

        df['day'] = np.ones(df.shape[0]).astype('int')
        df['month'] = month
        df['year'] = year

        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']])
        df.set_index(['datetime'], inplace=True)

        df_list.append(df)


    annual_df_merged = pd.concat(df_list,
                                 ignore_index=False)
    annual_df_merged.reset_index(inplace=True)

    annual_df_merged['dt_start'] = annual_df_merged.datetime
    annual_df_merged['dt_stop'] = annual_df_merged.datetime
    annual_df_merged['frp_mean'] = annual_df_merged.frp_x
    annual_df_merged['frp_std'] = annual_df_merged.frp_x

    grouped = annual_df_merged.groupby(['lats_x', 'lons_x']).agg({'times_seen_in_month': np.sum,
                                                              'dt_start': np.min,
                                                              'dt_stop': np.max,
                                                              'frp_mean': np.mean,
                                                              'frp_std': np.std})
    grouped.reset_index(inplace=True)
    if not os.path.exists(os.path.join(fp.path_to_test_csv_out, 'all_sensors')):
        os.makedirs(os.path.join(fp.path_to_test_csv_out, 'all_sensors'))
    grouped.to_csv(os.path.join(fp.path_to_test_csv_out, 'all_sensors', 'all_flares.csv'), date_format='%Y-%m-%d %H:%M:%S')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()