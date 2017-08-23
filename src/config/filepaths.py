'''
Holds the filepaths for the project
'''

import src.config.constants as constants

# path to ggf code
ggf_root_path = ''
ggf_dir = ggf_root_path + ''

# Paths to ATSR data on CEMS
if constants.sensor == 'ats':
    path_to_data = '/neodc/aatsr_multimission/aatsr-v3/data/ats_toa_1p'
elif constants.sensor == 'at2':
    path_to_data = '/neodc/aatsr_multimission/atsr2-v3/data/at2_toa_1p'
elif constants.sensor == 'at1':
    path_to_data = '/neodc/aatsr_multimission/atsr1-v3/data/at1_toa_1p'


# Paths to processing output folder on CEMS
output_root_path = ''
path_to_output = output_root_path + constants.sensor + '/'


# data analysis paths


# Testing paths on mac
root = '/Users/danielfisher/Projects/kcl-globalgasflaring/data/'
path_to_aatsr_test_data = root + 'raw/atsr/'
path_to_atsr2_test_data = ''
path_to_atsr1_test_data = ''
path_to_landcover_test = root + 'raw/landwater/land_water_mask_reduced_mode_900m.nc'
path_to_output_test = root + 'processed/ats/'

