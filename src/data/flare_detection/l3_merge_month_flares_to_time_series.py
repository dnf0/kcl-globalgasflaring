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
                array_list.append(v)                
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


def generate_coords(df):
    return zip(df.lats.values, df.lons.values)


def main():

    df_list = []
    csv_files = glob.glob(os.path.join(fp.path_to_cems_output_l3, "*", '*', "*_flaring_subset.csv"))

    for f in csv_files:
        df = pd.read_csv(f)

        df['day'] = np.ones(df.shape[0]).astype('int')
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']])
        df.set_index(['datetime'], inplace=True)

        df_list.append(df)


    all_data_df = pd.concat(df_list,
                                 ignore_index=False)
    all_data_df.reset_index(inplace=True)

    all_data_df['dt_start'] = all_data_df.datetime
    all_data_df['dt_stop'] = all_data_df.datetime
    all_data_df.rename(columns={'times_seen_in_month': 'times_seen'}, inplace=True)
    all_data_df['frp_sd'] = all_data_df.frp

    grouped_df = all_data_df.groupby(['lats', 'lons', 'sensor'], as_index=False).agg({'times_seen': np.sum,
                                                                                    'dt_start': np.min,
                                                                                    'dt_stop': np.max,
                                                                                    'frp': frp_median,
                                                                                    'frp_sd': frp_sd})
    grouped_df['coords'] = generate_coords(grouped_df)
    unique_coords = grouped_df.coords.unique()
    coords_df = pd.DataFrame({'coords': unique_coords,
                              'flare_id': np.arange(unique_coords.size)})
    grouped_df = grouped_df.merge(coords_df, how='left', on='coords')
    grouped_df.drop(['coords'], axis=1, inplace=True)

    if not os.path.exists(os.path.join(fp.path_to_cems_output_l3, 'all_sensors')):
        os.makedirs(os.path.join(fp.path_to_cems_output_l3, 'all_sensors'))
    grouped_df.to_csv(os.path.join(fp.path_to_cems_output_l3, 'all_sensors', 'all_flares.csv'),
                   date_format='%Y-%m-%d %H:%M:%S',
                   index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
