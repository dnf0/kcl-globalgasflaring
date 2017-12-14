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

# Paths to processing output folder on CEMS
path_to_cems_output_l2 = '/home/users/dnfisher/nceo_aerosolfire/data/temp/processed/l2/'
path_to_cems_output_l3 = '/home/users/dnfisher/nceo_aerosolfire/data/temp/processed/l3/'

###########  LOCAL PATHS (Data Analysis) ###########

local_root = '/Users/danielfisher/Projects/kcl-globalgasflaring/'
#local_root = '/Users/dnf/Projects/kcl-globalgasflaring/'

# paths for calculating in-band solar irradiance
path_to_solar_data = local_root + 'data/external/solar_irradiance/SOLAR_IN.txt'
path_to_srfs = local_root + 'data/external/spectral_response_funtions'


