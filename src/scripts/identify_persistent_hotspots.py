import os
import sys
import glob
import pandas as pd
import numpy as np

import src.config.filepaths as fp


def load_csvs(paths, cols=None) -> pd.DataFrame:
    """
    Generate a dataframe from a set of CSV files retaining
    specified columns.

    Args:
        paths: List of csv files
        cols: Columns to use (all if None)

    Returns:
        Pandas dataframe generated from the input CSV files
    """
    df_container = []
    for p in paths:
        try:
            df_container.append(pd.read_csv(p, usecols=cols))
        except pd.errors.EmptyDataError:
            continue
    return pd.concat(df_container, ignore_index=True)


def orbits_to_months(df, subset_cols=None) -> pd.DataFrame:
    """
    Reduce orbital level data to a monthly product that
    flags arcminute resolution grid cells with at least
    one detection in any given month.
    Args:
        df: dataframe with all orbit information
        subset_cols: columns to use in evaluation

    Returns:
        Reduced dataframe indiciating grid cells with at least
        one hotspot detection in any given month.

    """
    if subset_cols is None:
        subset_cols = ['grid_x', 'grid_y', 'year', 'month']
    return df.drop_duplicates(subset=subset_cols)


def months_to_annual_counts(df):
    """
    Given a dataframe containing monthly detections returns
    an aggregated dataframe providing the total number of hotspot
    detections counted in a given gridcell over an approximate
    twelve month period.

    Args:
        df: Monthly detection dataframe

    Returns:
        Aggregated dataframe of annum hotspot counts

    """
    df['day'] = 1  # arbitrary day column to allow use of pd.to_datetime
    df['dt'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['counter'] = 1

    annual_count_df = None

    # iterate over unique datetimes
    for start_dt in sorted(df['dt'].unique()):

        # create approximate 12 month period (~365 days)
        stop_dt = start_dt + pd.to_timedelta(365, unit='days')

        # subset and aggregate
        sub_df = df[start_dt <= df.dt < stop_dt]
        sub_df = sub_df.groupby(['grid_x', 'grid_y']).agg({'counter': np.sum})

        # append
        if annual_count_df is None:
            annual_count_df = sub_df
        else:
            annual_count_df = annual_count_df.append(sub_df, ignore_index=True)

    return annual_count_df


def main():

    sensor = sys.argv[1]
    if sensor not in ['atx', 'sls']:
        raise KeyError("Sensor not in" + "['atx', 'sls']")

    # set paths and target columns
    if sensor == 'atx':
        paths = glob.glob(fp.atx_hotspots)
        cols = ['grid_x', 'grid_y', 'year', 'month']
        min_count = 4
    else:
        paths = glob.glob(fp.sls_hotspots)
        cols = ['grid_x', 'grid_y', 'year', 'month']
        min_count = 2

    df = load_csvs(paths, cols=cols)
    df = orbits_to_months(df)
    df = months_to_annual_counts(df)
    df = df[df.counter > min_count]
    df.to_csv(os.path.join(fp.output_l3, sensor + f"all_flare_locations_{sensor}.csv"))


if __name__ == "__main__":
    main()
