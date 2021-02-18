#!/apps/jasmin/jaspy/miniconda_envs/jaspy3.7/m3-4.6.14/envs/jaspy3.7-m3-4.6.14-r20200606/bin/python3

import os
import sys
import epr
import pandas as pd
import numpy as np

from src.ggf.detectors import ATXDetector, SLSDetector
import src.utils as utils
import src.config.filepaths as fp


def merge_hotspot_dataframes(atx_persistent_df, sls_persistent_df):
    atx_persistent_df['sensor'] = 1
    sls_persistent_df['sensor'] = -1
    appended_df = atx_persistent_df.append(sls_persistent_df)
    hotspot_df = appended_df.groupby(['lats_arcmin', 'lons_arcmin'], as_index=False).agg({'sensor': np.sum})

    # the below information is not used here but we can use this
    # idea to update the sensor information during the post processing
    hotspot_df.sensor.loc[hotspot_df.sensor == 1] = 'atx'
    hotspot_df.sensor.loc[hotspot_df.sensor == -1] = 'sls'
    hotspot_df.sensor.loc[hotspot_df.sensor == 0] = 'both'
    return hotspot_df


def aggregate(df, aggregator):
    return df.groupby(['grid_y', 'grid_x'], as_index=False).agg(aggregator)


def main():
    file_to_process = sys.argv[1]
    sensor = sys.argv[2]

    if sensor != 'sls':
        product = epr.Product(file_to_process)
        HotspotDetector = ATXDetector(product)

        flare_keys = ['latitude',
                      'longitude',
                      'local_cloudiness',
                      'swir_16',
                      'frp',
                      'pixel_size',
                      'mwir',
                      'background_mwir']

        flare_aggregator = {'frp': np.sum,
                            'swir_16': np.mean,
                            'mwir': np.mean,
                            'background_mwir': np.mean,
                            'pixel_size': np.sum,
                            'latitude': np.mean,
                            'longitude': np.mean,
                            'local_cloudiness': np.mean}

        sampling_keys = ['latitude',
                         'longitude',
                         'local_cloudiness']

        sampling_aggregator = {'local_cloudiness': np.mean}

        atx_persistent_fp = os.path.join(fp.output_l3,
                                         'all_sensors',
                                         'all_flare_locations_ats.csv')
        persistent_df = pd.read_csv(atx_persistent_fp)

    else:
        product = utils.extract_zip(file_to_process, fp.slstr_extract_temp)
        HotspotDetector = SLSDetector(product)

        flare_keys = ['latitude',
                      'longitude',
                      'local_cloudiness',
                      'swir_16',
                      'swir_22',
                      'frp',
                      'pixel_size']

        flare_aggregator = {'frp': np.sum,
                            'swir_16': np.mean,
                            'swir_22': np.mean,
                            'pixel_size': np.sum,
                            'latitude': np.mean,
                            'longitude': np.mean,
                            'local_cloudiness': np.mean}

        sampling_keys = ['latitude',
                         'longitude',
                         'local_cloudiness']

        sampling_aggregator = {'local_cloudiness': np.mean}

        # merge persistent dataframes for SLSTR
        atx_persistent_fp = os.path.join(fp.output_l3,
                                         'all_sensors',
                                         'all_flare_locations_ats.csv')
        atx_persistent_df = pd.read_csv(atx_persistent_fp)

        sls_persistent_fp = os.path.join(fp.output_l3,
                                         'all_sensors',
                                         'all_flare_locations_sls.csv')
        sls_persistent_df = pd.read_csv(sls_persistent_fp)

        persistent_df = merge_hotspot_dataframes(atx_persistent_df,
                                                 sls_persistent_df)
    # find persistent hotspots (i.e. flares)
    HotspotDetector.run_detector(flares_or_sampling=True)
    flare_df = HotspotDetector.to_dataframe(keys=flare_keys,
                                            joining_df=persistent_df)
    aggregated_flare_df = aggregate(flare_df, flare_aggregator)
    aggregated_flare_df.to_csv(utils.build_outpath(sensor, file_to_process, 'flares'))

    # get sampling associated with persistent hotspots
    sampling_df = HotspotDetector.to_dataframe(keys=sampling_keys,
                                               joining_df=persistent_df)
    aggregated_sampling_df = aggregate(sampling_df, sampling_aggregator)
    aggregated_sampling_df.to_csv(utils.build_outpath(sensor, file_to_process, 'sampling'))


if __name__ =="__main__":
    main()
