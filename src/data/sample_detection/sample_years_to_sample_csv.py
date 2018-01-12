import logging
import os
import glob

import pandas as pd

import src.config.filepaths as fp


def main():

    # set up output path
    path_to_out = os.path.join(fp.path_to_cems_output_l3, 'all_sensors')

    annual_csv_filepaths = glob.glob(path_to_out + '*_sampling.csv')
    df_list = []
    for f in annual_csv_filepaths:
        if f.split('/')[-1] == 'all_sampling.csv':
            continue
        df_list.append(pd.read_csv(f))
    output_df = pd.concat(df_list, ignore_index=True)

    # dump to csv
    output_df.to_csv(os.path.join(path_to_out, 'all_sampling.csv'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()