#!/home/users/dnfisher/soft/virtual_envs/ggf/bin/python2

import os
import glob
import logging

import pandas as pd

import src.config.filepaths as fp


def main():

    df_list = []

    csv_filepaths = glob.glob(fp.path_to_cems_output_l2 + '*/*/*/*/*_sampling.csv')
    for f in csv_filepaths:
        if 'S3A' in f:
            continue
        print f
        df_list.append(pd.read_csv(f))

    df = pd.concat(df_list, ignore_index=True)

    path_to_out = os.path.join(fp.path_to_cems_output_l3, 'all_sensors')
    if not os.path.exists(path_to_out):
        os.makedirs(path_to_out)
    df.to_csv(os.path.join(path_to_out, 'all_sampling_atx_adaptive.csv'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
