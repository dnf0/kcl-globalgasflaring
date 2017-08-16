'''
This file contains some constants used in the processing
'''
import scipy.constants as const

ats_solar_irradiance = 254.484  # W m-2 sr-1 um-1  see email from Dave Smith
at2_solar_irradiance = 254.484  # W m-2 sr-1 um-1  this needs updating
at1_solar_irradiance = 254.484  # W m-2 sr-1 um-1  this needs updating

day_night_angle = -5  # 0 less than this angle is considered night

# thresholds for masking out clouds over sea
vis_thresh = 10  # percent
tir_thresh = 290  # kelvin

# number of standard deviations which constitutes a glas flare
n_std = 4

# frp calculation parameters
atsr_pixel_size = 1000000  # metres

ats_planck_approx = 99999 # TODO define this
at2_planck_approx = 99999 # TODO define this
at1_planck_approx = 99999 # TODO define this

ats_frp_coeff = const.sigma / ats_planck_approx
at2_frp_coeff = const.sigma / at2_planck_approx
at1_frp_coeff = const.sigma / at1_planck_approx