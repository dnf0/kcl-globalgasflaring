'''
Holds the filepaths for the project
'''


###########  JASMIN PATHS ###########

jasmin_root = '/home/users/dnfisher/projects/kcl-globalgasflaring/'

# path to ggf code
script_temp = jasmin_root + 'src/scripts/'
slstr_extract_temp = jasmin_root + 'src/data/'

# Paths to ATSR data on CEMS (setup for recursive glob searching)
products = {'ats': '/neodc/aatsr_multimission/aatsr-v3/data/ats_toa_1p/**/*.N1',
            'at2': '/neodc/aatsr_multimission/atsr2-v3/data/at2_toa_1p/**/*.E2',
            'at1': '/neodc/aatsr_multimission/atsr1-v3/data/at1_toa_1p/**/*.E1',
            'sls': '/neodc/sentinel3a/data/SLSTR/L1_RBT/**/*.h5'}

# Paths to processing output folder on JASMIN
output_root = '/group_workspaces/jasmin2/nceo_aerosolfire/data/temp/'
output_l2 = output_root + 'processed/l2/'
output_l3 = output_root + 'processed/l3/'

# Paths for product searching during data aggregation steps (setup for recursive glob searching)
# TODO naming of stages put into constants to ensure consistency
atx_hotspots = output_l2 + '**/*AT*hotspots.csv'
sls_hotspots = output_l2 + '**/*S3*hotspots.csv'
atx_flares = output_l2 + '**/*AT*flares.csv'
sls_flares = output_l2 + '**/*S3*flares.csv'
atx_sampling = output_l2 + '**/*AT*samples.csv'
sls_sampling = output_l2 + '**/*S3*samples.csv'

# TODO slurm logging paths
slurm_info = ""
slurm_error = ""

###########  LOCAL PATHS (Data Analysis) ###########

local_root = '/Users/danielfisher/Projects/kcl-globalgasflaring/'
#local_root = '/Users/dnf/Projects/kcl-globalgasflaring/'

# paths for calculating in-band solar irradiance
path_to_solar_data = local_root + 'data/external/solar_irradiance/SOLAR_IN.txt'
path_to_srfs = local_root + 'data/external/spectral_response_funtions'


