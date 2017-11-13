import os
import glob
import logging

import pandas as pd
import numpy as np

import src.config.filepaths as fp


def main():

    for sensor in ['ats', 'at1', 'at2']:

        orbit_line_sample = []

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
                for f in csv_files_for_month:
                    try:
                        df = pd.read_csv(f)
                    except Exception, e:
                        logger.warning('Could not load csv file with error: ' + str(e))

                    orbit = [o.split('_')[-1] for o in df.fname.values()]
                    orbit_line_sample.append(zip(orbit, df.lines.values, df.sample.values))

            # after each year reduce to only unique
            orbit_line_sample = list(set(orbit_line_sample))

        orbit, line, sample = zip(*orbit_line_sample)
        ols_df = pd.DataFrame({'orbit': orbit, 'line': line, 'sample': sample})
        path_to_out = os.path.join(fp.path_to_test_csv_out, sensor)
        if not os.path.exists(path_to_out):
            os.makedirs(path_to_out)
        ols_df.to_csv(os.path.join(path_to_out, sensor + '_ols.csv'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()