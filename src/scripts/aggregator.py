import pandas as pd
import numpy as np


def extract_dataframes(paths, cols=None) -> pd.DataFrame:
    """
    Generate a dataframe from a set of CSV files retaining
    specified columns.

    Args:
        paths: List of csv files
        cols: Columns to use (all if None)

    Returns:
        Pandas dataframe generated from the input CSV files
    """
    return pd.concat([pd.read_csv(p, usecols=cols) for p in paths])


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

