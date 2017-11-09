'''
Merges all the monthly files for a given sensor into
a complete time series of gas flaring.
'''

import logging
import glob
import os

import numpy as np
import pandas as pd

import src.config.filepaths as fp


def main():

    # define sensor
    sensor = 'at1'

    # get all csv files for sensor
    root = os.path.join(fp.path_to_test_csv_out, sensor)
    csv_filepaths = glob.glob(root + '/*/*.csv')

    # assign 'time' id to csv files
    n_files = np.arange(len(csv_filepaths))

    # set up some things for the iteration
    time_series_df_list = []

    # iterate over csv files year blocks
    for start_file_index in n_files[:-12]:

        # get the df files for this block
        annual_df_files = csv_filepaths[start_file_index:start_file_index+12]

        # merge dataframes, but also insert the unique months into them first
        annual_df_list = []
        for i, f in enumerate(annual_df_files):
            split_f = f.split('/')
            year = split_f[-2]
            month = split_f[-1][0:2]
            month_df = pd.read_csv(f)
            month_df['month'] = np.ones(month_df.shape[0]) * int(month)
            month_df['year'] = np.ones(month_df.shape[0]) * int(year)
            annual_df_list.append(month_df)
        annual_df = pd.concat(annual_df_list, ignore_index=True)

        # find gas flares
        lat_lon_tuples = zip(annual_df.lats, annual_df.lons)
        unique_locations, indicies, unique_counts = np.unique(lat_lon_tuples, axis=0, return_inverse=True,
                                                              return_counts=True)
        annual_df['counts'] = unique_counts[indicies]
        annual_flare_df = annual_df[annual_df['counts'] >= 4]

        # append to time series dataframe
        time_series_df_list.append(annual_flare_df)

    # merge the time series
    time_series_df = pd.concat(time_series_df_list, ignore_index=True)

    # drop repeated gas flare observations
    lat_lon_time_tuples = zip(time_series_df.lats,
                              time_series_df.lons,
                              time_series_df.month,
                              time_series_df.year)

    # dump out the unique time series csv
    unique_locations, index = np.unique(lat_lon_time_tuples, return_index=True, axis=0)
    unique_time_series_df = time_series_df.iloc[index]
    unique_time_series_df.to_csv(os.path.join(root, 'time_series.csv'))


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()