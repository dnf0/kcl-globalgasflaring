#!/apps/jasmin/jaspy/miniconda_envs/jaspy3.7/m3-4.6.14/envs/jaspy3.7-m3-4.6.14-r20200606/bin/python3

import sys
import epr

from src.ggf.detectors import ATXDetector, SLSDetector
import src.utils as utils
import src.config.filepaths as fp


def main():
    file_to_process = sys.argv[1]
    sensor = sys.argv[2]

    if sensor != 'sls':
        product = epr.Product(file_to_process)
        HotspotDetector = ATXDetector(product)
        keys = ['latitude', 'longitude']
    else:
        product = utils.extract_zip(file_to_process, fp.slstr_extract_temp)
        HotspotDetector = SLSDetector(product)
        keys = ['latitude', 'longitude', 'sza', 'vza', 'swir_16', 'swir_22']

    HotspotDetector.run_detector()
    df = HotspotDetector.to_dataframe(keys=keys)
    df.to_csv(utils.build_outpath(sensor, file_to_process, 'hotspots'))


if __name__ =="__main__":
    main()
