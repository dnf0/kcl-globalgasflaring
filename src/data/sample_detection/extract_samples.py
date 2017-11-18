import os
import glob
import logging

import pandas as pd
import numpy as np

from collections import defaultdict

import src.config.filepaths as fp


def main():

    sample_counter = defaultdict(int)
    lats = defaultdict(float)
    lons = defaultdict(float)

    for sensor in ['ats', 'at1', 'at2']:
        year_dir = os.path.join(fp.path_to_test_csv, sensor)
        years = os.listdir(year_dir)
        for year in years:
            month_dir = os.path.join(year_dir, year)
            try:
                months = os.listdir(month_dir)
            except:
                continue
            for month in months:
                csv_files_for_month = glob.glob(os.path.join(fp.path_to_test_csv,
                                                            sensor,
                                                            year,
                                                            month,
                                                            "*", "*_sampling.csv"))
                for f in csv_files_for_month:
                    try:
                        sample_df = pd.read_csv(f)
                        for index, row in sample_df.iterrows():
                            sample_counter[row.flare_ids] += 1
                            lats[row.flare_ids] = row.matched_lats
                            lons[row.flare_ids] = row.matched_lons

                    except Exception, e:
                        logger.warning('Could not load csv file with error: ' + str(e))


    out_df = pd.DataFrame({"sample_counts": sample_counter,
                           "sample_lats": lats,
                           "sample_lons": lons})

    # dump to csv
    path_to_out = os.path.join(fp.path_to_test_csv_out)
    out_df.to_csv(os.path.join(path_to_out, 'sampling.csv'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
