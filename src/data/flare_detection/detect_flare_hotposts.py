'''
Merges all the monthly files for a given sensor into
a complete time series of gas flaring.

There are two different approaches, a monthly one and annual one.

Algorithm is as follows:




'''

import logging
import glob
import os
import re

import numpy as np
import pandas as pd

import src.config.filepaths as fp

def generate_monthly_dataframes(root):
    monthly_df_fnames = []
    years = np.arange(1991,2013,1)
    months = np.arange(1,13,1)
    for yr in years:
        for m in months:
            csv_files = glob.glob(os.path.join(root, '*', str(yr), str(m).zfill(2) + '.csv'))
            if len(csv_files) == 1:
                df = setup_df(csv_files[0])
            elif len(csv_files) == 2:
                # in combined dataframe we will be getting more observations, this is dealt with
                # in construct annual df by dropping duplicated locations.
                df_a = setup_df(csv_files[0])
                df_b = setup_df(csv_files[1])
                df = df_a.append(df_b, ignore_index=True)
            else:
                continue
            out_path = os.path.join(fp.path_to_cems_output_intermediate, str(yr) + str(m).zfill(2) + '.csv')
            df.to_csv(out_path, index=False)
            monthly_df_fnames.append(out_path)
    return monthly_df_fnames


def generate_years_months_iterator():
    iterator = []
    years = np.arange(1991,2013,1)
    months = np.arange(1,13,1)
    for yr in years:
        for m in months:
            iterator.append((yr, m))
    return iterator



def construct_annual_df(root, year_month_subset):
    annual_df_list = []

    # iterate over each month and read in csv files
    for y, m in year_month_subset:

        csv_files = glob.glob(os.path.join(root, '*', str(y), str(m).zfill(2) + '.csv'))

        # sometimes we can have observations from AT1 and AT2 or AT2 and ATS, so might be two csv files.
        # Each sensor might have differing locations, so need to keep that in consideration and evaluate both.
        if len(csv_files) == 1:
            df = pd.read_csv(csv_files[0], usecols=['lats', 'lons'], dtype={'lats': float, 'lons': float})
        elif len(csv_files) == 2:
            df_a = pd.read_csv(csv_files[0], usecols=['lats', 'lons'], dtype={'lats': float, 'lons': float})
            df_b = pd.read_csv(csv_files[1], usecols=['lats', 'lons'], dtype={'lats': float, 'lons': float})
            df = df_a.append(df_b, ignore_index=True)

        # drop any duplicate locations for the month so that we only get a single observations per location per month
        # this deals with any overlaps between sensors.
        df.drop_duplicates(inplace=True)

        # append to annual dataframe list
        annual_df_list.append(df)

    # in some cases (at the start of the time series) the annual dataframe will have less than 12 observations
    # meaning that we wil miss some flaring locations.  As we will not have a full set of 12 months.  Just
    # need to make sure that this is made clear in any outputs.
    annual_df = pd.concat(annual_df_list, ignore_index=True)
    annual_df['times_seen_in_annum'] = np.ones(annual_df.shape[0])
    return annual_df


def group_annual_df(annual_df):
    return annual_df.groupby(['lats', 'lons'], as_index=False).agg({'times_seen_in_annum': np.sum})


def detect_persistent_hotspots(grouped_annual_df):
    return grouped_annual_df[grouped_annual_df['times_seen_in_annum'] >= 4]


def main():

    # get all csv files for sensor
    root = os.path.join(fp.path_to_cems_output_l3)

    # make an iterator to loop over all years
    years_months = generate_years_months_iterator()

    # using the monthly dataframes perform the flare detection
    all_flares_df = None
    for i in len(years_months[:-12]):

        # read in the required dataframes
        annual_df = construct_annual_df(root, years_months[i:i+12])

        # group the annual dataframe by lats and lons
        grouped_annual_df = group_annual_df(annual_df)

        # get annual hotspots
        annual_hotspot_df = detect_persistent_hotspots(grouped_annual_df)

        if all_flares_df is None:
            all_flares_df = annual_hotspot_df
        else:
            all_flares_df = all_flares_df.append(annual_hotspot_df, ignore_index=True)

            # keep only the unique flaring locations in all flares, giving a map
            # of all flare locations seen over the entire ATSR time series.
            all_flares_df.drop_duplicates(inplace=True)
        all_flares_df.to_csv(os.path.join(fp.path_to_cems_output_l3, 'all_sensors', 'all_flare_locations.csv'))

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()