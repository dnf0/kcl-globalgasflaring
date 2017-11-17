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

    # iterate over csv files year blocks
    for start_file_index in n_files[:-12]:

        # get the df files for this block
        annual_df_files = csv_filepaths[start_file_index:start_file_index + 12]

        # read the twelve months from the current month, retain the current month dataframe
        annual_df_list = []
        for i, f in enumerate(annual_df_files):

            if i == 0:
                current_month_df = pd.read_csv(f)
                month_flare_out_path = f.replace('.csv', '_flaring_subset.csv')
                annual_flare_out_path = f.replace('.csv', '_flaring_subset_annual.csv')

            annual_df_list.append(pd.read_csv(f))

        # concatenate the monthly dataframes
        annual_df = pd.concat(annual_df_list, ignore_index=True)

        # add in useful groupby metrics
        annual_df['times_seen_in_annum'] = np.ones(annual_df.shape[0])
        annual_df['lat_std'] = annual_df['lat']
        annual_df['lon_std'] = annual_df['lon']
        annual_df['frp_std'] = annual_df['frp']
        annual_df['mean_monthly_times_seen'] = annual_df['times_seen_in_month']
        annual_df['std_monthly_times_seen'] = annual_df['times_seen_in_month']

        # now group the annual dataframe by lats and lons
        grouped_annual_df = annual_df.groupby(['lats', 'lons']).agg({'frp': np.median,
                                                                     'frp_std': np.std,
                                                                     'lats': np.mean,
                                                                     'lons': np.mean,
                                                                     'lats_std': np.std,
                                                                     'lons_std': np.std,
                                                                     'times_seen_in_annum': np.sum,
                                                                     'mean_monthly_times_seen': np.mean,
                                                                     'std_monthly_times_seen': np.std,
                                                                     })

        # now retain only those locations with flaring activity (i.e. count >= 4)
        grouped_annual_df = grouped_annual_df[grouped_annual_df['times_seen_in_annum'] >= 4]

        # now save the annual flaring characteristics
        grouped_annual_df.to_csv(annual_flare_out_path)

        # now subset the month to only valid flaring locations do this by merging on lats and lons
        current_month_df.merge(grouped_annual_df, on=['lats', 'lons'])

        # now save the monthly dataframe
        current_month_df.to_csv(month_flare_out_path)

    # the months from n_files[:-12] have not been processed, do them now using the last annual dataframe
    for f in csv_filepaths[-12:]:
        month_flare_out_path = f.replace('.csv', '_flaring_subset.csv')
        current_month_df = pd.read_csv(f)
        current_month_df.merge(grouped_annual_df, on=['lats', 'lons'])
        current_month_df.to_csv(month_flare_out_path)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
