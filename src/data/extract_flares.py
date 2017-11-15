import os
import glob
import logging
from datetime import datetime

import pandas as pd
import numpy as np

import src.config.filepaths as fp


def main():

    root = fp.path_to_test_csv_out
    at1_monthly_timeseries = pd.read_csv(os.path.join(root, 'at1', 'monthly_time_series.csv'))
    at2_monthly_timeseries = pd.read_csv(os.path.join(root, 'at2', 'monthly_time_series.csv'))
    ats_monthly_timeseries = pd.read_csv(os.path.join(root, 'ats', 'monthly_time_series.csv'))

    for df in [at1_monthly_timeseries, at2_monthly_timeseries, ats_monthly_timeseries]:
        df.drop('Unnamed: 0', 1, inplace=True)
        df['number_of_flares_in_month'] = np.ones(df.shape[0])
        df['day'] = np.ones(df.shape[0]).astype('int')
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']])
        df.set_index(['datetime'], inplace=True)
        df.drop(['year', 'month', 'day'], 1, inplace=True)

    at1_monthly_timeseries = at1_monthly_timeseries[(at1_monthly_timeseries.index >= datetime(1991, 8, 1)) &
                                                    (at1_monthly_timeseries.index < datetime(1996, 6, 1))]
    at2_monthly_timeseries = at2_monthly_timeseries[(at2_monthly_timeseries.index >= datetime(1996, 6, 1)) &
                                                    (at2_monthly_timeseries.index < datetime(2003, 1, 1))]
    ats_monthly_timeseries = ats_monthly_timeseries[(ats_monthly_timeseries.index >= datetime(2003, 1, 1))]
    annual_df_merged = pd.concat([at1_monthly_timeseries, at2_monthly_timeseries, ats_monthly_timeseries],
                                 ignore_index=False)
    annual_df_merged.reset_index(inplace=True)

    annual_df_merged['dt_start'] = annual_df_merged.datetime
    annual_df_merged['dt_stop'] = annual_df_merged.datetime
    annual_df_merged['frp_mean'] = annual_df_merged.frp
    annual_df_merged['frp_std'] = annual_df_merged.frp
    annual_df_merged['total_observations'] = annual_df_merged.number_of_flares_in_month

    grouped = annual_df_merged.groupby(['lats', 'lons']).agg({'total_observations': np.sum,
                                                              'dt_start': np.min,
                                                              'dt_stop': np.max,
                                                              'frp_mean': np.mean,
                                                              'frp_std': np.std})
    grouped.reset_index(inplace=True)
    if not os.path.exists(root):
        os.makedirs(root)
    grouped.to_csv(os.path.join(root, 'all_flares.csv'), date_format='%Y-%m-%d %H:%M:%S')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()