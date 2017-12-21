'''
Holds the filepaths for the project
'''

import src.config.constants as constants


###########  CEMS PATHS ###########

cems_root = '/home/users/dnfisher/projects/kcl-globalgasflaring/'

# path to ggf code
ggf_dir = cems_root + 'src/ggf/'

# Paths to ATSR data on CEMS
paths_to_data = ['/neodc/aatsr_multimission/aatsr-v3/data/ats_toa_1p',
                 '/neodc/aatsr_multimission/atsr2-v3/data/at2_toa_1p',
                 '/neodc/aatsr_multimission/atsr1-v3/data/at1_toa_1p']

# Paths to processing output folder on CEMS
output_root = '/home/users/dnfisher/nceo_aerosolfire/data/temp/'
#output_root = '/Users/dnf/Projects/kcl-globalgasflaring/data/'
#output_root = '/Users/danielfisher/Projects/kcl-globalgasflaring/data/'
path_to_cems_output_intermediate = output_root + 'processed/l3/intermediate/'
path_to_cems_output_l2 = output_root + 'processed/l2/'
path_to_cems_output_l3 = output_root + 'processed/l3/'

###########  LOCAL PATHS (Data Analysis) ###########

local_root = '/Users/danielfisher/Projects/kcl-globalgasflaring/'
#local_root = '/Users/dnf/Projects/kcl-globalgasflaring/'

# paths for calculating in-band solar irradiance
path_to_solar_data = local_root + 'data/external/solar_irradiance/SOLAR_IN.txt'
path_to_srfs = local_root + 'data/external/spectral_response_funtions'


