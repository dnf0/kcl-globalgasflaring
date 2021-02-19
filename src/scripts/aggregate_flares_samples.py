import os
import glob
import pandas as pd

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


def main():

    roots = [fp.atx_flares, fp.atx_sampling, fp.sls_flares, fp.sls_sampling]
    csv_names = ['atx_flares', 'atx_sampling', 'sls_flares', 'sls_sampling']
    for r, csv_name in zip(roots, csv_names):
        paths = glob.glob(r)
        df = load_csvs(paths)
        df.to_csv(os.path.join(fp.output_l3, f"{csv_name}.csv"))


if __name__ == "__main__":
    main()
