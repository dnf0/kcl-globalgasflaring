'''
This file contains some constants used in the processing
'''
import scipy.constants as const

null_value = -999

min_background_proportion = 0.6  # fraction

solar_irradiance = {'ats': 254.752,
                    'at2': 249.604,
                    'at1': 250.728}

# sza > 101 deg VIIRS Elvidge et al., 2017
day_night_angle = 101  # degrees

# flare detection limit, based on min value in ATS.
atx_swir_threshold = 0.06  # W m2 sr-1 um-1

# set radiance thresholds (pixel are factor of four
# smaller, so increase by factor of 4 to make similar)
sls_swir_threshold = 4 * atx_swir_threshold  # W m2 sr-1 um-1
sls_vza_threshold = 22  # degrees

# cloud statistics computation window size
atx_cloud_window_size = 17  # pixels
sls_cloud_window_size = 33  # pixels

# mwir background window size
atx_background_window_size = 17  # pixels

# fire radiative power caluclation coefficients
frp_coeff = {'sls': const.sigma / 8.19919059044e-09,
             'ats': const.sigma / 8.29908189231e-09,
             'at2': const.sigma / 8.2215268253e-09,
             'at1': const.sigma / 8.23565040885e-09}
