import os
import glob
import logging

import pandas as pd

import src.config.filepaths as fp

def main():

    for sensor in ['ats', 'at1', 'at2']:
        years = os.listdir(os.path.join(fp.path_to_test_csv, sensor))
        for year in years:
            print sensor, year
            csv_files_for_year = glob.glob(os.path.join(fp.path_to_test_csv,
                                                        sensor,
                                                        year,
                                                        "*/*", "*.csv"))
            df_list = []
            for f in csv_files_for_year:
                try:
                    df_list.append(pd.read_csv(f))
                except Exception, e:
                    logger.warning('Could not load csv file with error: ' + str(e))

            df_for_year = pd.concat(df_list, ignore_index=True)
            df_for_year.to_csv(os.path.join(fp.path_to_test_csv_out, sensor + '_' + year + '.csv'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
