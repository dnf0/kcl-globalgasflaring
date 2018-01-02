import os
import glob
import logging

import pandas as pd
import numpy as np

import src.config.filepaths as fp


def check_file(fname):
    ymd = fname[14:22]
    y = int(ymd[0:4])
    m = int(ymd[4:6])

    if 'AT2' in fname:
        if (y == 2000) & (m == 12):
            return True
        elif (y == 2001) & (m >= 1) & (m <= 6):
            return True
        elif (y == 2001) & (m >= 9) & (m <= 12):
            return True
        elif (y == 2002) & (m >= 1) & (m <= 6):
            return True
        else:
            return False

    elif 'ATS' in fname:
        if (y == 2002) & (m == 5):
            return True
        else:
            return False

    else:
        return False


def main():

    csv_filepaths = glob.glob(fp.path_to_cems_output_l2 + '*/*/*/*/*_sampling.csv')
    output_df = None
    to_group = ['lats_arcmin', 'lons_arcmin']
    agg_dict = {'sample_counts': np.sum,
                'cloud_free_counts': np.sum,
                'flare_counts': np.sum,
                }

    # now lets get count the samples
    for f in csv_filepaths:
        try:
            # check if yr and month of csv file are in permitted months
            fname = f.split('/')[-1]
            if check_file(fname):
                print 'not processing f', f
                continue
            sample_df = pd.read_csv(f)

            # add in sample, cloud free and flare counts columns
            sample_df['sample_counts'] = 1.
            sample_df['cloud_free_counts'] = (sample_df.types == 2).astype(int)
            sample_df['flare_counts'] = (sample_df.types == 1).astype(int)

            # group samples to nearest arc minute, which gives the individual flares
            # and sum total obs (flares + cloud free), total flares, and total cloud free
            grouped_sample_df = sample_df.groupby(to_group, as_index=False).agg(agg_dict)

            if output_df is None:
                output_df = grouped_sample_df
            else:
                # merge each csv df to the total df, which records the total number of samples
                # across entire time series, excluding the months we are not interested in
                output_df = output_df.append(grouped_sample_df)
                output_df = output_df.groupby(to_group, as_index=False).agg(agg_dict)

        except Exception, e:
            logger.warning('Could not load csv file with error: ' + str(e))

    # dump to csv
    if not os.path.exists(os.path.join(fp.path_to_cems_output_l3, 'all_sensors')):
        os.makedirs(os.path.join(fp.path_to_cems_output_l3, 'all_sensors'))

    output_df.to_csv(os.path.join(fp.path_to_cems_output_l3, 'all_sensors', 'all_sampling.csv'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
