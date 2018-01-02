# Using the flare data produced in the flare detection
# module determine the start and stop time of each
# detected gas flare so that persistency can be accurately
# determined.

import os
import logging

import pandas as pd
import numpy as np

import src.config.filepaths as fp

def main():
    cols = ['frp', 'radiances', 'reflectances', 'sun_elev',
            'view_elev', 'pixel_size', 'lats', 'lons', 'lats_arcmin',
            'lons_arcmin', 'year', 'month', 'day', 'hhmm', 'sensor',
            'se_dist', 'frp_coeff']
    dtypes = {'frp': float, 'radiances': float, 'reflectances': float, 'sun_elev': float,
            'view_elev': float, 'pixel_size': float, 'lats': float, 'lons': float, 'lats_arcmin': int,
            'lons_arcmin': int, 'year': int, 'month': int, 'day': int, 'hhmm': int, 'sensor': str,
            'se_dist': float, 'frp_coeff': float}

    path_to_out = os.path.join(fp.path_to_cems_output_l3, 'all_sensors')
    all_flares_path = os.path.join(path_to_out, 'all_flares.csv')
    flare_df = pd.read_csv(all_flares_path, usecols=cols, dtype=dtypes)

    # first set up an single ID for the flare based on location
    cols = ['lats_arcmin', 'lons_arcmin']
    index_df = flare_df.drop_duplicates(['lats_arcmin', 'lons_arcmin']).reset_index()[cols]
    index_df['flare_id'] = index_df.index

    # apply to flare df so that we get indexes
    indexed_flare_df = pd.merge(flare_df, index_df, on=['lats_arcmin', 'lons_arcmin'])

    # datetimes (we are working on a daily sampling level)
    indexed_flare_df['dt'] = pd.to_datetime(indexed_flare_df['year'].astype(str) + ' '
                                            + indexed_flare_df['month'].astype(str) + ' '
                                            + indexed_flare_df['day'].astype(str))
    indexed_flare_df['dt_start'] = indexed_flare_df['dt']
    indexed_flare_df['dt_stop'] = indexed_flare_df['dt']

    # grouped by datetime and flare id and get mean frp for each day (need to group and flares
    # might be seen in overlapping orbits on the same day, or by two sensors on the same day)
    grouped_dt = indexed_flare_df.groupby(cols, as_index=False).agg({'dt_start': np.min, 'dt_stop': np.max})

    if not os.path.exists(path_to_out):
        os.makedirs(path_to_out)
    grouped_dt.to_csv(os.path.join(path_to_out, 'flare_start_stop.csv'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()