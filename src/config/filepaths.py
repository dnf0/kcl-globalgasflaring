'''
Holds the filepaths for the project
'''

import src.config.constants as constants


###########  CEMS PATHS ###########

cems_root = '/home/users/dnfisher/projects/kcl-globalgasflaring/'

# path to ggf code
ggf_dir = cems_root + 'src/ggf/'

# Paths to ATSR data on CEMS
if constants.sensor == 'ats':
    path_to_data = '/neodc/aatsr_multimission/aatsr-v3/data/ats_toa_1p'
elif constants.sensor == 'at2':
    path_to_data = '/neodc/aatsr_multimission/atsr2-v3/data/at2_toa_1p'
elif constants.sensor == 'at1':
    path_to_data = '/neodc/aatsr_multimission/atsr1-v3/data/at1_toa_1p'

# path to flare df on cems
path_to_flare_df = ''

# Paths to processing output folder on CEMS
#path_to_cems_output = cems_root + 'data/processed/' + constants.sensor + '/'
path_to_cems_output = '/home/users/dnfisher/nceo_aerosolfire/data/temp/' + constants.sensor 

###########  LOCAL PATHS ###########

local_root = '/Users/danielfisher/Projects/kcl-globalgasflaring/'
#local_root = '/Users/dnf/Projects/kcl-globalgasflaring/'

# paths for calculating in-band solar irradiance
path_to_solar_data = local_root + 'data/external/solar_irradiance/SOLAR_IN.txt'
path_to_srfs = local_root + 'data/external/spectral_response_funtions'

# data analysis paths


# Testing paths on mac
path_to_test_csv = local_root + '/data/processed'
path_to_test_csv_out = local_root + 'data/processed/merged_months'

path_to_aatsr_test_data = local_root + 'raw/atsr/'
path_to_atsr2_test_data = ''
path_to_atsr1_test_data = ''
path_to_landcover_test = local_root + 'raw/landwater/land_water_mask_reduced_mode_900m.nc'
path_to_output_test = local_root + 'processed/ats/'

