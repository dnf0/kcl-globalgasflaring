'''
Scans sls temp dir and removes any empty
files
'''

import logging
import os

import src.config.filepaths as fp


def main():
    while True:
        for f in os.listdir(fp.path_to_temp):
            print 'removing', f
            try:
                os.rmdir(f)
            except:
                continue


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
