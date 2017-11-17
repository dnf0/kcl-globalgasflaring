"""
Merge gas flare detections to monthly samples.

The first step in the algorithm is to reduce the resolution of the data:

1. Iterate over all detection files,
2. Adjust the lats, lons to the specified resolution
3. Reduce the flares based on the adjusted lats and lons, taking the mean.
4. Append the reduced flares to the dataframe

This dataframe will give all the monthly observations.  We need to take the mean for the
orbit, as this will give a proper representation of the flare over the cluster.


The second step in the algorithm is to aggregate the reduced resolution data to monthly
2. Append a counter to the dataframe
3. Reduce by lats and lons, taking the median frp, and the count, also record lats and lons.
 Taking the median over the month is better as we can assume that the flare burns consistently
 and that the majority of changes are caused by variation in cloud cover.  Taking the median
 will give the cloud free result assuming there is optically thin cloud < 60% of the time.

"""

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

                        orbit_df = pd.read_csv(f)
                        orbit_df['lons'] = myround(orbit_df['lons'].values, base=resolution)
                        orbit_df['lats'] = myround(orbit_df['lats'].values, base=resolution)
                        orbit_df = orbit_df.groupby(['lons', 'lats']).agg({'frp': np.mean,
                                                                        'lats': np.mean, 'lons': np.mean})
                        orbit_df.reset_index(inplace=True)

                        month_flares.append(orbit_df)
                    except Exception, e:
                        logger.warning('Could not load csv file with error: ' + str(e))

                month_df = pd.concat(month_flares, ignore_index=True)

                # add in ones for counting number of flare obs in months
                month_df['times_seen_in_month'] = np.ones(month_df.shape[0])

                # compute the mean FRP
                month_df['lats_std'] = month_df['lats']
                month_df['lons_std'] = month_df['lons']
                month_df['frp_std'] = month_df['frp']
                month_df = month_df.groupby(['lats', 'lons']).agg({'frp': np.median, 'frp_std': np.std,
                                                                   'lats': np.mean, 'lons': np.mean,
                                                                   'lats_std': np.std, 'lons_std': np.std,
                                                                   'times_seen_in_month': np.sum})
                month_df.reset_index(inplace=True)

                # dump to csv
                path_to_out = os.path.join(fp.path_to_test_csv_out, sensor, year)
                if not os.path.exists(path_to_out):
                    os.makedirs(path_to_out)
                month_df.to_csv(os.path.join(path_to_out, month + '.csv'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()