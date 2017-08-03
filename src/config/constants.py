'''
This file contains some constants used in the processing
'''
import scipy.constants as const

solar_irradiance = 254.484  # W m-2 sr-1 um-1  see email from Dave Smith
day_night_angle = 0  # less than this angle is considered night

# thresholds for masking out clouds over sea
vis_thresh = 10  # percent
tir_thresh = 290  # kelvin

# number of standard deviations which constitutes a glas flare
n_std = 4

# frp calculation parameters
atsr_pixel_size = 1000000  # metres
atsr_planck_approx = 99999  # TODO define this
frp_coeff = const.sigma / atsr_planck_approx