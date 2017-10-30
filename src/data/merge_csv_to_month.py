import os
import glob
import logging

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

import src.config.filepaths as fp


def main():

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
                                                            "*", "*.csv"))
                month_flares = []
                for f in csv_files_for_month:
                    try:
                        month_flares.append(pd.read_csv(f))
                    except Exception, e:
                        logger.warning('Could not load csv file with error: ' + str(e))

                df_for_month = pd.concat(month_flares, ignore_index=True)

                # cluster for the month
                coords = df_for_month.as_matrix(columns=['lats', 'lons'])
                kms_per_radian = 6371.0088
                epsilon = 1 / kms_per_radian
                db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(
                    np.radians(coords))

                # assign the clusters
                df_for_month['clusters'] = db.labels_

                # compute the mean FRP TODO extent this to other values
                df_for_month = df_for_month.groupby('clusters').agg({'frp': np.mean, 'lats': np.mean, 'lons': np.mean})

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
