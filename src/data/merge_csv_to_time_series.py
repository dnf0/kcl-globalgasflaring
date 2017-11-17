'''
Merges all the monthly files for a given sensor into
a complete time series of gas flaring.

There are two different approaches, a monthly one and annual one.

Algorithm is as follows:
1. Set the current month
2. Read in 12 months worth of monthly data from the current month appending the year and month and counter to each.
3. Add useful reduction paramters to the dataframe
4. Reduce by lats and lons and count the number of times each lat lon combination appears
5. Retain only those flares appear more than minimum sampling
6. Save that dataframe as the annual flaring characteristics during the 12 months from the current month
7. Inner join the annual and monthly dataframes on the lats and lons, keeping only those flares in the current
 month that were observed at least 4 times over the year.
8. Deal with last twelve months



'''

import logging
import glob
import os
import re

import numpy as np
import pandas as pd

import src.config.filepaths as fp


def main():
    # define sensor
    sensor = 'ats'

    # get all csv files for sensor
    root = os.path.join(fp.path_to_test_csv_out, sensor)
    csv_filepaths = glob.glob(root + '/*/*.csv')

    # filter the csv filepaths to just get the raw month files (as sometimes we might have other outputs in here)
    csv_filepaths = [f for f in csv_filepaths if re.search(r'[0-9]+.csv', f)]

    # assign 'time' id to csv files
    n_files = np.arange(len(csv_filepaths))

    # make annual df list to hold set of 12 annual dataframes
    list_of_grouped_annual_dfs = []

    # iterate over csv files year blocks
    for start_file_index in n_files[:-12]:

        # get the df files for this block
        annual_df_files = csv_filepaths[start_file_index:start_file_index + 12]

        # read the twelve months from the current month, retain the current month dataframe
        annual_df_list = []
        for i, f in enumerate(annual_df_files):

            if i == 0:

                split_f = f.split('/')
                year = split_f[-2]
                month = split_f[-1][0:2]

                current_month_df = pd.read_csv(f)
                month_flare_out_path = f.replace('.csv', '_flaring_subset.csv')
                annual_flare_out_path = f.replace('.csv', '_flaring_subset_annual.csv')

            annual_df_list.append(pd.read_csv(f))

        # concatenate the monthly dataframes
        annual_df = pd.concat(annual_df_list, ignore_index=True)

        # add in useful groupby metrics
        annual_df['times_seen_in_annum'] = np.ones(annual_df.shape[0])
        annual_df['frp_std'] = annual_df['frp']
        annual_df['mean_monthly_times_seen'] = annual_df['times_seen_in_month']
        annual_df['std_monthly_times_seen'] = annual_df['times_seen_in_month']

        # now group the annual dataframe by lats and lons
        grouped_annual_df = annual_df.groupby(['lats', 'lons']).agg({'frp': np.median,
                                                                     'frp_std': np.std,
                                                                     'lats': np.mean,
                                                                     'lons': np.mean,
                                                                     'times_seen_in_annum': np.sum,
                                                                     'mean_monthly_times_seen': np.mean,
                                                                     'std_monthly_times_seen': np.std,
                                                                     })

        # now retain only those locations with flaring activity (i.e. count >= 4)
        grouped_annual_df = grouped_annual_df[grouped_annual_df['times_seen_in_annum'] >= 4]

        # we need to do a moving detector over the current month, and we need to look from 12 months
        # before to 12 months after to get all possible flares that might be burning during the current month.
        # So lets store 12 sets annums.
        list_of_grouped_annual_dfs.append(grouped_annual_df)
        if len(list_of_grouped_annual_dfs) == 12:
            list_of_grouped_annual_dfs.pop(0)
        grouped_annual_dfs = pd.concat(list_of_grouped_annual_dfs, ignore_index=True)

        # now group again by lat and lon to reduce to the unique flaring locations in the 12 annums
        # and reset index to get the lats/lons back out.
        regrouped_annual_dfs = grouped_annual_dfs.groupby(['lats', 'lons'], as_index=False).agg({'frp': np.median})

        # now subset the month to only valid flaring locations do this by merging on lats and lons
        # but first we need to create a combined column of lats and lons in the set of 12 annums
        regrouped_annual_dfs['coords'] = zip(regrouped_annual_dfs.lats.values, regrouped_annual_dfs.lons.values)
        current_month_df['coords'] = zip(current_month_df.lats.values, current_month_df.lons.values)

        # we only want to keep coordinates from the regrouped annual dataframe
        regrouped_annual_dfs = regrouped_annual_dfs['coords']

        # now do the merge
        current_month_df = current_month_df.merge(regrouped_annual_dfs, on=['coords'])

        # add in the year and the month to the dataframes to save doing it elsewhere
        grouped_annual_df['year'] = year
        grouped_annual_df['month'] = month
        current_month_df['year'] = year
        current_month_df['month'] = month

        # now save the monthly and annual dataframes
        grouped_annual_df.to_csv(annual_flare_out_path)
        current_month_df.to_csv(month_flare_out_path)

    # the months from n_files[:-12] have not been processed, do them now using the last annual dataframe
    for f in csv_filepaths[-12:]:

        split_f = f.split('/')
        year = split_f[-2]
        month = split_f[-1][0:2]

        month_flare_out_path = f.replace('.csv', '_flaring_subset.csv')
        current_month_df = pd.read_csv(f)
        current_month_df['coords'] = zip(current_month_df.lats.values, current_month_df.lons.values)
        current_month_df = current_month_df.merge(regrouped_annual_dfs, on=['coords'])
        current_month_df['year'] = year
        current_month_df['month'] = month
        current_month_df.to_csv(month_flare_out_path)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
