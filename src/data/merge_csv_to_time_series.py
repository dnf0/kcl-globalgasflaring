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
    time_series_list = []

    # iterate over csv files year blocks
    for start_file_index in n_files[:-12]:

        # get the df files for this block
        annual_df_files = csv_filepaths[start_file_index:start_file_index+12]

        # merge dataframes, but also insert the unique months into them first
        annual_hotspot_list = []
        for i, f in enumerate(annual_df_files):
            month_hotspot_df = pd.read_csv(f) # open up the monthly dataframe
            month_hotspot_df.drop('cluster_ids', 1, inplace=True)  # drop monthly cluster_id as no longer relevant
            annual_hotspot_list.append(month_hotspot_df)  # append an annual list (i.e. 12 months of data)

            # add in month and year
            split_f = f.split('/')
            year = split_f[-2]
            month = split_f[-1][0:2]
            month_hotspot_df['month'] = np.ones(month_hotspot_df.shape[0]) * int(month)
            month_hotspot_df['year'] = np.ones(month_hotspot_df.shape[0]) * int(year)

            if i == 0:
                current_month = int(month)

        # join the 12 months of files into one dataframe
        annual_hotspot_df = pd.concat(annual_hotspot_list, ignore_index=True)

        # now restrict to only those hotspots seen at least four times in twelve months
        lat_lon_tuples = zip(annual_hotspot_df.lats, annual_hotspot_df.lons)
        _, cluster_ids, unique_counts = np.unique(lat_lon_tuples, axis=0, return_inverse=True, return_counts=True)

        # now we search across the 12 months to find all hotspots that have been seen
        # at least 4 times during this time period.  The unique_counts gives us the number
        # of times each hotspot appears.  The cluster_ids assign the count to the right hotspot
        # e.g. counts = [3,1,1], ids = [0,0,0,1,2] counts[ids] = [3,3,3,1,1], we just select
        # all those instances where counts is greater than some threshol
        annual_hotspot_df['times_seen_in_twelve_months'] = unique_counts[cluster_ids]
        annual_flare_df = annual_hotspot_df[annual_hotspot_df['times_seen_in_twelve_months'] >= 4]

        # now we want to restrict the annual counts to just our month.  We do this
        # by subsetting down to the current month.  This gives us the number of
        # flares seen in this month, that were also observed at least 4 times the next 12 months.
        month_flare_df = annual_flare_df[annual_flare_df.month == current_month]

        # append to month flaring dataframe
        time_series_list.append(month_flare_df)

    # merge the time series
    time_series_df = pd.concat(time_series_list, ignore_index=True)
    time_series_df.to_csv(os.path.join(root, 'monthly_time_series.csv'))

    # example code to get number of unique flares
    lat_lon_tuples = zip(time_series_df.lats, time_series_df.lons)
    _, unique_counts = np.unique(lat_lon_tuples, axis=0, return_counts=True)
    print unique_counts.size


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()