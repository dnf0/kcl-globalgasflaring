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
    sensor = 'ats'

    # get all csv files for sensor
    root = os.path.join(fp.path_to_test_csv_out, sensor)
    csv_filepaths = glob.glob(root + '/*/*.csv')

    # assign 'time' id to csv files
    n_files = np.arange(len(csv_filepaths))

    # set up some things for the iteration
    time_series_list_monthly = []
    time_series_list_annual = []

    # # iterate over csv files year blocks
    # for start_file_index in n_files[:-12]:
    #
    #     # get the df files for this block
    #     annual_df_files = csv_filepaths[start_file_index:start_file_index+12]
    #
    #     # merge dataframes, but also insert the unique months into them first
    #     annual_hotspot_list = []
    #     for i, f in enumerate(annual_df_files):
    #         month_hotspot_df = pd.read_csv(f) # open up the monthly dataframe
    #         month_hotspot_df.drop('cluster_ids', 1, inplace=True)  # drop monthly cluster_id as no longer relevant
    #         annual_hotspot_list.append(month_hotspot_df)  # append an annual list (i.e. 12 months of data)
    #
    #         # add in month and year
    #         split_f = f.split('/')
    #         year = split_f[-2]
    #         month = split_f[-1][0:2]
    #         month_hotspot_df['month'] = np.ones(month_hotspot_df.shape[0]) * int(month)
    #         month_hotspot_df['year'] = np.ones(month_hotspot_df.shape[0]) * int(year)
    #
    #         if i == 0:
    #             current_month = int(month)
    #
    #     # join the 12 months of files into one dataframe
    #     annual_hotspot_df = pd.concat(annual_hotspot_list, ignore_index=True)
    #
    #     # now restrict to only those hotspots seen at least four times in twelve months
    #     lat_lon_tuples = zip(annual_hotspot_df.lats, annual_hotspot_df.lons)
    #     _, cluster_ids, unique_counts = np.unique(lat_lon_tuples, axis=0, return_inverse=True, return_counts=True)
    #
    #     # now we search across the 12 months to find all hotspots that have been seen
    #     # at least 4 times during this time period.  The unique_counts gives us the number
    #     # of times each hotspot appears.  The cluster_ids assign the count to the right hotspot
    #     # e.g. counts = [3,1,1], ids = [0,0,0,1,2] counts[ids] = [3,3,3,1,1], we just select
    #     # all those instances where counts is greater than some threshol
    #     annual_hotspot_df['times_seen_in_twelve_months'] = unique_counts[cluster_ids]
    #     annual_flare_df = annual_hotspot_df[annual_hotspot_df['times_seen_in_twelve_months'] >= 4]
    #
    #     # now we want to restrict the annual counts to just our month.  We do this
    #     # by subsetting down to the current month.  This gives us the number of
    #     # flares seen in this month, that were also observed at least 4 times the next 12 months.
    #     month_flare_df = annual_flare_df[annual_flare_df.month == current_month]
    #
    #     # append to month flaring dataframe
    #     time_series_list.append(month_flare_df)
    #
    # # merge the time series
    # time_series_df = pd.concat(time_series_list, ignore_index=True)
    # time_series_df.to_csv(os.path.join(root, 'monthly_time_series.csv'))

    # iterate over csv files year blocks
    for start_file_index in n_files[:-12]:

        # get the df files for this block
        annual_df_files = csv_filepaths[start_file_index:start_file_index + 12]

        # merge dataframes, but also insert the unique months into them first
        annual_df_list = []
        for i, f in enumerate(annual_df_files):
            split_f = f.split('/')
            year = split_f[-2]
            month = split_f[-1][0:2]
            month_df = pd.read_csv(f)
            month_df['month'] = (np.ones(month_df.shape[0]) * int(month)).astype('int')
            month_df['year'] = (np.ones(month_df.shape[0]) * int(year)).astype('int')
            annual_df_list.append(month_df)

            if i == 0:
                current_month = month
                current_year = year

        annual_df = pd.concat(annual_df_list, ignore_index=True)

        # find gas flares
        lat_lon_tuples = zip(annual_df.lats, annual_df.lons)
        unique_locations, indicies, unique_counts = np.unique(lat_lon_tuples, axis=0, return_inverse=True,
                                                              return_counts=True)
        annual_df['clusters'] = indicies
        annual_df['counts'] = unique_counts[indicies]
        annual_flare_df = annual_df[annual_df['counts'] >= 4]

        # append to time series for monthly data to dataframe
        time_series_list_monthly.append(annual_flare_df)

        # get the yearly counts for each month
        annual_grouped = annual_flare_df.groupby('clusters').agg({'frp': np.mean,
                                                                  'lats': np.mean,
                                                                  'lons': np.mean})
        annual_grouped['month'] = (np.ones(annual_grouped.shape[0]) * int(current_month)).astype('int')
        annual_grouped['year'] = (np.ones(annual_grouped.shape[0]) * int(current_year)).astype('int')
        time_series_list_annual.append(annual_grouped)


    # merge the time series
    time_series_df_monthly = pd.concat(time_series_list_monthly, ignore_index=True)
    time_series_df_annual = pd.concat(time_series_list_annual, ignore_index=True)

    # drop repeated gas flare observations
    lat_lon_time_tuples_monthly = zip(time_series_df_monthly.lats,
                                      time_series_df_monthly.lons,
                                      time_series_df_monthly.month,
                                      time_series_df_monthly.year)
    lat_lon_time_tuples_annual = zip(time_series_df_annual.lats,
                                     time_series_df_annual.lons,
                                     time_series_df_annual.month,
                                     time_series_df_annual.year)

    # dump out the unique time series csv
    unique_locations, index = np.unique(lat_lon_time_tuples_monthly, return_index=True, axis=0)
    unique_time_series_df = time_series_df_monthly.iloc[index]
    unique_time_series_df.to_csv(os.path.join(root, 'monthly_time_series.csv'))

    unique_locations, index = np.unique(lat_lon_time_tuples_annual, return_index=True, axis=0)
    unique_time_series_df = time_series_df_annual.iloc[index]
    unique_time_series_df.to_csv(os.path.join(root, 'annual_time_series.csv'))


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
