'''
Script to process VIIRS data to extract flaring
'''


import logging
import os

import h5py
import numpy as np
import pandas as pd

def get_filelist():
    pass


def read_data():
    pass


def detect_hotspots():
    pass


def characterise_hotspots():
    pass


def main():

    # get viirs files
    path_to_data = ''
    viirs_fnames = get_filelist(path_to_data)

    # set up output dataframe
    df_list = []

    # iterate over viirs files
    for f in viirs_fnames:

        ds = read_data(os.path.join(path_to_data, f))

        # extract hotspots

        # characterise hotspots

        # append to df

    # concat df

    # write out





if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
