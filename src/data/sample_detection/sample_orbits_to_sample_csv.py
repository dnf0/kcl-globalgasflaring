import os
import glob
import logging
from datetime import datetime

import pandas as pd
import numpy as np

import src.config.filepaths as fp


def check_file(fname, ymd):

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

    # set up output path
    path_to_out = os.path.join(fp.path_to_cems_output_l3, 'all_sensors')

    # flare start stop df
    start_stop_df = pd.read_csv(os.path.join(path_to_out, 'flare_start_stop.csv'))
    start_stop_df.dt_start = pd.to_datetime(start_stop_df.dt_start) 
    start_stop_df.dt_stop = pd.to_datetime(start_stop_df.dt_stop)

    output_df = None
    to_subset = ['lats_arcmin', 'lons_arcmin']
    to_group = ['lats_arcmin', 'lons_arcmin', 'year']
    agg_dict = {'sample_counts': np.sum,
                'cloud_free_counts': np.sum,
                'flare_counts': np.sum,
                }

    csv_filepaths = glob.glob(fp.path_to_cems_output_l2 + '*/*/*/*/*_sampling.csv')
    for f in csv_filepaths:
        try:
            # check if yr and month of csv file are in permitted months
            fname = f.split('/')[-1]
            ymd = fname[14:22]
            file_time = datetime.strptime(ymd, '%Y%m%d')
            if check_file(fname, ymd):
                print 'not processing f', f
                continue

            # read csv and add in new columns
            sample_df = pd.read_csv(f)
            sample_df['sample_counts'] = 1.
            sample_df['cloud_free_counts'] = (sample_df.types == 2).astype(int)
            sample_df['flare_counts'] = (sample_df.types == 1).astype(int)

            # reduce start stop dataframe to only those flares seen burning before of after
            # current csv file time.  Giving the indexes of flares actually operating during
            # this overpass of this csv file.
            valid_start_stop_df = start_stop_df[(start_stop_df.dt_start <= file_time) &
                                                (start_stop_df.dt_stop >= file_time)]

            # reduce the sample df to the operating flares by merging on coordinates
            sample_df = pd.merge(sample_df, valid_start_stop_df, on=to_subset)
            
            # record the samples in the output df, grouping each orbit csv as it is
            # appended and summing the sampling counts.  Resulting in each flare
            # grouped by year.
            if output_df is None:
                output_df = sample_df
                print output_df.head()
            else:
                output_df = output_df.append(sample_df)
                output_df = output_df.groupby(to_group, as_index=False).agg(agg_dict)
                print output_df.head()

        except Exception, e:
            logger.warning('Could not load csv file with error: ' + str(e))

    # dump to csv
    output_df.to_csv(os.path.join(path_to_out, 'all_sampling_annual.csv'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()