import os
import glob
import logging
import re
from collections import defaultdict

import pandas as pd

import src.config.filepaths as fp

def main():

    csv_filepaths = glob.glob(fp.path_to_cems_output_l2 + '*/*/*/*/*_sampling.csv')

    sample_counter = defaultdict(int)
    cloud_free_counts = defaultdict(int)
    flare_counts = defaultdict(int)
    lats = defaultdict(float)
    lons = defaultdict(float)
    lats_arcmin = defaultdict(int)
    lons_arcmin = defaultdict(int)

    # now lets get count the samples
    for f in csv_filepaths:
        try:
            sample_df = pd.read_csv(f)

            for index, row in sample_df.iterrows():

                k = str(row.lats_arcmin) + str(row.lons_arcmin)

                sample_counter[k] += 1
                lats[k] = row.lats
                lons[k] = row.lons
                lats_arcmin[k] = row.lats_arcmin
                lons_arcmin[k] = row.lons_arcmin
                if row.types == 2:
                    cloud_free_counts[k] += 1
                if row.types == 1:
                    flare_counts[k] += 1

        except Exception, e:
            logger.warning('Could not load csv file with error: ' + str(e))

    out_df = pd.DataFrame({"sample_counts": sample_counter,
                           "cloud_free_counts": cloud_free_counts,
                           "flare_counts": flare_counts,
                           "sample_lats": lats,
                           "sample_lons": lons,
                           "lats_arcmin": lats_arcmin,
                           "lons_arcmin": lons_arcmin})

    # dump to csv
    if not os.path.exists(os.path.join(fp.path_to_cems_output_l3, 'all_sensors')):
        os.makedirs(os.path.join(fp.path_to_cems_output_l3, 'all_sensors'))

    out_df.to_csv(os.path.join(fp.path_to_cems_output_l3, 'all_sensors', 'all_sampling.csv'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
