import os
import glob
import logging

import pandas as pd
import numpy as np

import src.config.filepaths as fp


def myround(x, dec=20, base=.000005):
    return np.round(base * np.round(x/base), dec)


def main():

    # aggregation resolution
    # resolution = 15. / 3600  # in arcseconds. 3600 arc seconds in a degree, and want nearest 15 arcseconds (~0.5km)
    resolution = 60. / 3600  # arseconds ~2km

    for sensor in ['ats', 'at1', 'at2']:
        year_dir = os.path.join(fp.path_to_test_csv, sensor)
        years = os.listdir(year_dir)
        for year in years:
            month_dir = os.path.join(year_dir, year)
            months = os.listdir(month_dir)
            for month in months:
                csv_files_for_month = glob.glob(os.path.join(fp.path_to_test_csv,
                                                            sensor,
                                                            year,
                                                            month,
                                                            "*", "*_flares.csv"))
                month_flares = []
                for f in csv_files_for_month:
                    try:
                        month_flares.append(pd.read_csv(f))
                    except Exception, e:
                        logger.warning('Could not load csv file with error: ' + str(e))

                df_for_month = pd.concat(month_flares, ignore_index=True)

                # cluster for the month by first rounding to the desired resolution
                # and then getting the set of unique locations.  Much faster than DB scan.
                df_for_month['lons'] = myround(df_for_month['lons'].values, base=resolution)
                df_for_month['lats'] = myround(df_for_month['lats'].values, base=resolution)
                lat_lon_tuples = zip(df_for_month.lats, df_for_month.lons)
                _, cluster_ids = np.unique(lat_lon_tuples, axis=0, return_inverse=True)

                df_for_month['cluster_ids'] = cluster_ids
                df_for_month['times_seen_in_month'] = np.ones(df_for_month.shape[0])

                # compute the mean FRP TODO extent this to other values
                df_for_month = df_for_month.groupby('cluster_ids').agg({'frp': np.mean,
                                                                     'lats': np.mean, 'lons': np.mean,
                                                                     'times_seen_in_month': np.sum})

                # dump to csv
                path_to_out = os.path.join(fp.path_to_test_csv_out, sensor, year)
                if not os.path.exists(path_to_out):
                    os.makedirs(path_to_out)
                df_for_month.to_csv(os.path.join(path_to_out, month + '.csv'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
