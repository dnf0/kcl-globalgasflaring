import os
import glob
import logging

import pandas as pd
import numpy as np

import src.config.filepaths as fp


def frp_median(s):

        array_list = []
        for v in s.values:
            try:
                l = np.matrix(v.strip("[ ]")).tolist()  # TODO super hacky, need to find a better way
                array_list += l[0]
            except Exception, e:
                print (e)
                print v
                print
                continue
        return np.median(array_list)


def frp_sd(s):
    try:
        array_list = []
        for v in s.values:
            l = np.matrix(v.strip("[ ]")).tolist()
            array_list += l[0]
        return np.std(array_list)
    except:
        return 99999


def main():

    df_list = []
    csv_files = glob.glob(os.path.join(fp.path_to_cems_output_l3, "*", '*', "*_flaring_subset.csv"))

    for f in csv_files:
        df = pd.read_csv(f)

        df['day'] = np.ones(df.shape[0]).astype('int')
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']])
        df.set_index(['datetime'], inplace=True)

        df_list.append(df)


    merged_df = pd.concat(df_list,
                                 ignore_index=False)
    merged_df.reset_index(inplace=True)

    merged_df['dt_start'] = merged_df.datetime
    merged_df['dt_stop'] = merged_df.datetime
    merged_df.rename(columns={'times_seen_in_month': 'times_seen'}, inplace=True)
    merged_df['frp_sd'] = merged_df.frp

    grouped = merged_df.groupby(['lats', 'lons'], as_index=False).agg({'times_seen': np.sum,
                                                                       'dt_start': np.min,
                                                                       'dt_stop': np.max,
                                                                       'frp': frp_median,
                                                                       'frp_sd': frp_sd})

    if not os.path.exists(os.path.join(fp.path_to_cems_output_l3, 'all_sensors')):
        os.makedirs(os.path.join(fp.path_to_cems_output_l3, 'all_sensors'))
    grouped.to_csv(os.path.join(fp.path_to_cems_output_l3, 'all_sensors', 'all_flares.csv'),
                   date_format='%Y-%m-%d %H:%M:%S',
                   index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()