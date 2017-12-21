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


def get_sensor(path):
    if 'ats' in path:
        return 'ats'
    elif 'at2' in path:
        return 'at2'
    elif 'at1' in path:
        return 'at1'


def setup_df(path, yr, m):
    sensor = get_sensor(path)
    print 'in setup_df, sensor is:', sensor
    df = pd.read_csv(path)
    df['sensor'] = sensor
    df['year'] = yr
    df['month'] = m
    return df


def generate_monthly_dataframes(root):
    monthly_df_fnames = []
    years = np.arange(1991,2013,1)
    months = np.arange(1,13,1)
    for yr in years:
        for m in months:
            csv_files = glob.glob(os.path.join(root, '*', str(yr), str(m).zfill(2) + '.csv'))
            if len(csv_files) == 1:
                df = setup_df(csv_files[0], yr, m)
            elif len(csv_files) == 2:
                df_a = setup_df(csv_files[0], yr, m)
                df_b = setup_df(csv_files[1], yr, m)
                df = df_a.append(df_b, ignore_index=True)
            else:
                continue
            out_path = os.path.join(fp.path_to_cems_output_intermediate, str(yr) + str(m).zfill(2) + '.csv')
            df.to_csv(out_path, index=False)
            monthly_df_fnames.append(out_path)
    return monthly_df_fnames



def generate_coords(df):
    return zip(df.lats.values, df.lons.values)


def construct_annual_df(df_files_for_annum):
    annual_df_list = []
    for f in df_files_for_annum:
        annual_df_list.append(pd.read_csv(f, usecols=['lats', 'lons']))

    # concatenate the monthly dataframes
    annual_df = pd.concat(annual_df_list, ignore_index=True)
    annual_df['times_seen_in_annum'] = np.ones(annual_df.shape[0])
    return annual_df


def group_annual_df(annual_df):
    return annual_df.groupby(['lats', 'lons'], as_index=False).agg({'times_seen_in_annum': np.sum})


def detect_persistent_hotspots(grouped_annual_df):
    return grouped_annual_df[grouped_annual_df['times_seen_in_annum'] >= 4]


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


def save(month_df, annual_df, root):

    for sensor in ['ats', 'at2', 'at1']:
        month_df = month_df[month_df.sensor == sensor]
        if not month_df.empty:
            print 'in save, and df not empoy, saving for sensor:', sensor
            out_path = os.path.join(root, sensor, str(month_df.year[0]), str(month_df.month[0]).zfill(2))
            month_out_path = out_path + '_flaring_subset.csv'
            annual_out_path = out_path + '_flaring_subset_annual.csv'

            month_df.to_csv(month_out_path, index=False)
            annual_df.to_csv(annual_out_path, index=False)


def save_month(month_df, root):

    for sensor in ['ats', 'at2', 'at1']:
        month_df = month_df[month_df.sensor == sensor]
        if not month_df.empty:
            out_path = os.path.join(root, sensor, str(month_df.year[0]), str(month_df.month[0]).zfill(2))
            month_out_path = out_path + '_flaring_subset.csv'
            month_df.to_csv(month_out_path, index=False)


def main():

    # get all csv files for sensor
    root = os.path.join(fp.path_to_cems_output_l3)

    # extract all the monthly dataframes and keep track of the
    # sensor that observed the month
    monthly_df_fnames = generate_monthly_dataframes(root)

    # using the monthly dataframes perform the flare detection
    list_of_12_hotspot_dfs = []
    for i, month_df_fname in enumerate(monthly_df_fnames[:-12]):

        # read in the required dataframes
        month_df = pd.read_csv(month_df_fname)
        annual_df = construct_annual_df(monthly_df_fnames[i:i+12])

        # add in coords
        month_df['coords'] = generate_coords(month_df)

        # group the annual dataframe by lats and lons
        grouped_annual_df = group_annual_df(annual_df)

        # get annual hotspots
        annual_hotspot_df = detect_persistent_hotspots(grouped_annual_df)

        # append current annum to the 12 annum hotspot location series
        _12_annum_hotspot_location_series = generate_12_annum_hotspot_df(list_of_12_hotspot_dfs, annual_hotspot_df)

        # now do the merge so that we get all flares detected over all overlapping 12 month periods
        # for the current month
        month_hotspot_df = month_df.merge(_12_annum_hotspot_location_series.to_frame(), on=['coords'])

        # split the detections by sensors and save the outputs
        save(month_hotspot_df, annual_hotspot_df, root)

    # the months from n_files[:-12] have not been processed, do them now using the last annual dataframe
    for month_df_fname in monthly_df_fnames[-12:]:
        month_df = pd.read_csv(month_df_fname)
        month_df['coords'] = generate_coords(month_df)
        month_hotspot_df = month_df.merge(_12_annum_hotspot_location_series.to_frame(), on=['coords'])
        save_month(month_hotspot_df, root)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
