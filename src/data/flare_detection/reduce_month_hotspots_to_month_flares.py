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


def get_year_month(f):
    split_f = f.split('/')
    year = split_f[-2]
    month = split_f[-1][0:2]
    return year, month


def construct_annual_df(df_files_for_annum):
    annual_df_list = []
    for i, f in enumerate(df_files_for_annum):
        annual_df_list.append(pd.read_csv(f, usecols=['lats', 'lons']))

    # concatenate the monthly dataframes
    annual_df = pd.concat(annual_df_list, ignore_index=True)
    return annual_df


def extend_annual_df(annual_df):
    annual_df['times_seen_in_annum'] = np.ones(annual_df.shape[0])


def group_annual_df(annual_df):
    return annual_df.groupby(['lats', 'lons'], as_index=False).agg({'times_seen_in_annum': np.sum})


def detect_persistent_hotspots(grouped_annual_df):
    return grouped_annual_df[grouped_annual_df['times_seen_in_annum'] >= 4]


def generate_coords(df):
    return zip(df.lats.values, df.lons.values)


def generate_12_annum_hotspot_df(list_of_12_hotspot_dfs, annual_hotspot_df):
    # we need to do a moving detector over the current month, and we need to look from 12 months
    # before to 12 months after to get all possible flares that might be burning during the current month.
    # So lets store 12 sets annums.
    list_of_12_hotspot_dfs.append(annual_hotspot_df)
    if len(list_of_12_hotspot_dfs) == 12:
        list_of_12_hotspot_dfs.pop(0)
    _12_hotspot_dfs = pd.concat(list_of_12_hotspot_dfs, ignore_index=True)

    # now subset the month to only valid flaring locations do this by merging on lats and lons
    # but first we need to create a combined column of lats and lons in the set of 12 annums
    _12_hotspot_dfs['coords'] = generate_coords(_12_hotspot_dfs)

    # we only want to keep coordinates from the regrouped annual dataframe
    _12_hotspot_dfs = _12_hotspot_dfs['coords']

    # reduce down to unique coords
    _12_hotspot_dfs.drop_duplicates(inplace=True)
    return _12_hotspot_dfs


def add_year_month_to_df(df, year, month):
    df['year'] = year
    df['month'] = month


def process_missing_months(f, _12_annum_hotspot_location_series):

    # get output
    month_flare_out_path = f.replace('.csv', '_flaring_subset.csv')

    # read in df and setup coords varialbe
    current_month_df = pd.read_csv(f)
    current_month_df['coords'] = generate_coords(current_month_df)

    # get hotspots for month
    month_hotspot_df = current_month_df.merge(_12_annum_hotspot_location_series.to_frame(), on=['coords'])

    # and year month data
    year, month = get_year_month(f)
    month_hotspot_df['year'] = year
    month_hotspot_df['month'] = month

    # save df
    month_hotspot_df.to_csv(month_flare_out_path, index=False)


def main():

    # get all csv files for sensor
    root = os.path.join(fp.path_to_test_csv_out)
    csv_filepaths = glob.glob(root + '/*/*/*.csv')

    # filter the csv filepaths to process
    csv_filepaths = [f for f in csv_filepaths if re.search(r'[0-9]+.csv', f)]
    csv_filepaths = [f for f in csv_filepaths if not re.search(r'at2/2002/[0][5-9].csv', f)]
    csv_filepaths = [f for f in csv_filepaths if not re.search(r'at2/2002/[1][0-2].csv', f)]
    csv_filepaths = [f for f in csv_filepaths if not re.search(r'at2/2003/[0-9]+.csv', f)]
    csv_filepaths = [f for f in csv_filepaths if not re.search(r'at1/199[6-7]/[0-9]+.csv', f)]
    csv_filepaths = [f for f in csv_filepaths if not re.search(r'at1/1995/[0][6-9].csv', f)]
    csv_filepaths = [f for f in csv_filepaths if not re.search(r'at1/1995/[1][0-2].csv', f)]

    # assign 'time' id to csv files
    n_files = np.arange(len(csv_filepaths))

    # make annual df list to hold set of 12 annual dataframes of hotspot locations
    list_of_12_hotspot_dfs = []

    # iterate over csv files year blocks
    for start_file_index in n_files[:-12]:

        # get the df files for this block
        df_file_for_current_month = csv_filepaths[start_file_index]
        df_files_for_annum = csv_filepaths[start_file_index:start_file_index + 12]

        # setup the output paths:
        month_flare_out_path = df_file_for_current_month.replace('.csv', '_flaring_subset.csv')
        annual_flare_out_path = df_file_for_current_month.replace('.csv', '_flaring_subset_annual.csv')

        # setup current month df
        current_month_df = pd.read_csv(df_file_for_current_month)
        current_month_df['coords'] = generate_coords(current_month_df)

        # read the twelve months from the current month, retain the current month dataframe
        annual_df = construct_annual_df(df_files_for_annum)

        # add in useful groupby metrics
        extend_annual_df(annual_df)

        # now group the annual dataframe by lats and lons
        grouped_annual_df = group_annual_df(annual_df)

        # get the annual hotspots
        annual_hotspot_df = detect_persistent_hotspots(grouped_annual_df)

        # append current annum to the 12 annum hotspot location series
        _12_annum_hotspot_location_series = generate_12_annum_hotspot_df(list_of_12_hotspot_dfs, annual_hotspot_df)

        # now do the merge so that we get all flares detected over all overlapping 12 month periods
        # for the current month
        month_hotspot_df = current_month_df.merge(_12_annum_hotspot_location_series.to_frame(), on=['coords'])

        # add in the year and the month to the dataframes to save doing it elsewhere
        year, month = get_year_month(df_file_for_current_month)
        add_year_month_to_df(annual_hotspot_df, year, month)
        add_year_month_to_df(month_hotspot_df, year, month)

        # now save the monthly and annual dataframes
        annual_hotspot_df.to_csv(annual_flare_out_path, index=False)
        month_hotspot_df.to_csv(month_flare_out_path, index=False)

    # the months from n_files[:-12] have not been processed, do them now using the last annual dataframe
    for df_file_for_current_month in csv_filepaths[-12:]:
        process_missing_months(df_file_for_current_month,
                               _12_annum_hotspot_location_series)



if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
