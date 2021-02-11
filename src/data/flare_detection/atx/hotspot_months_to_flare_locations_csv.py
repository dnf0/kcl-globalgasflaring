#!/apps/contrib/jaspy/miniconda_envs/jaspy2.7/m2-4.6.14/envs/jaspy2.7-m2-4.6.14-r20190715/bin/python2.7

'''

'''

import logging
import glob
import os
import re

import numpy as np
import pandas as pd

import src.config.filepaths as fp


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
        print(y, m)
        csv_files = glob.glob(os.path.join(root, '*', str(y), str(m).zfill(2) + '.csv'))

        # sometimes we can have observations from AT1 and AT2 or AT2 and ATS, so might be two csv files.
        # Each sensor might have differing locations, so need to keep that in consideration and evaluate both.
        if len(csv_files) == 0:
            continue
        elif len(csv_files) == 1:
            df = pd.read_csv(csv_files[0], usecols=['lats_arcmin', 'lons_arcmin'],
                             dtype={'lats_arcmin': int, 'lons_arcmin': int})
        elif len(csv_files) == 2:
            df_a = pd.read_csv(csv_files[0], usecols=['lats_arcmin', 'lons_arcmin'],
                               dtype={'lats_arcmin': int, 'lons_arcmin': int})
            df_b = pd.read_csv(csv_files[1], usecols=['lats_arcmin', 'lons_arcmin'],
                               dtype={'lats_arcmin': int, 'lons_arcmin': int})
            df = df_a.append(df_b, ignore_index=True)

        # drop any duplicate locations for the month so that we only get a single observations per location per month
        # this deals with any overlaps between sensors.
        df.drop_duplicates(inplace=True)

        # append to annual dataframe list
        annual_df_list.append(df)

    # in some cases (at the start of the time series) the annual dataframe will have less than 12 observations
    # meaning that we wil miss some flaring locations.  As we will not have a full set of 12 months.  Just
    # need to make sure that this is made clear in any outputs.
    print(len(annual_df_list))
    annual_df = pd.concat(annual_df_list, ignore_index=True)
    annual_df['times_seen_in_annum'] = np.ones(annual_df.shape[0])
    return annual_df


def group_annual_df(annual_df):
    return annual_df.groupby(['lats_arcmin', 'lons_arcmin'], as_index=False).agg({'times_seen_in_annum': np.sum})


def detect_persistent_hotspots(grouped_annual_df):
    return grouped_annual_df[grouped_annual_df['times_seen_in_annum'] >= 2]


def main():

    # get all csv files for sensor
    root = os.path.join(fp.path_to_cems_output_l3)

    # make an iterator to loop over all years
    years_months = generate_years_months_iterator()

    # using the monthly dataframes perform the flare detection
    all_flares_df = None
    for i in range(len(years_months[:-12])):
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
            all_flares_df.drop_duplicates(subset=['lats_arcmin', 'lons_arcmin'], inplace=True)
        print(all_flares_df.shape)
   
    path_to_out = os.path.join(fp.path_to_cems_output_l3, 'all_sensors')
    if not os.path.exists(path_to_out):
        os.makedirs(path_to_out)
    all_flares_df.to_csv(os.path.join(path_to_out, 'all_flare_locations_ats.csv'))

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
