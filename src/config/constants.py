'''
This file contains some constants used in the processing
'''
import scipy.constants as const

sensor = 'at2'  # 'at2', 'at1'

if sensor == 'ats':
    solar_irradiance = 254.484  # W m-2 sr-1 um-1  see email from Dave Smith
elif sensor == 'at2':
    solar_irradiance = 254.484  # W m-2 sr-1 um-1  this needs updating
elif sensor == 'at1':
    solar_irradiance = 254.484  # W m-2 sr-1 um-1  this needs updating

day_night_angle = -5  # 0 less than this angle is considered night

# thresholds for masking out clouds over sea
vis_thresh = 10  # percent
tir_thresh = 290  # kelvin

# number of standard deviations which constitutes a glas flare
n_std = 4

# frp calculation parameters
atsr_pixel_size = 1000000  # metres


if sensor == 'ats':
    planck_approx = 99999  # TODO define this
elif sensor == 'at2':
    planck_approx = 99999  # TODO define this
elif sensor == 'at1':
    planck_approx = 99999  # TODO define this

frp_coeff = const.sigma / planck_approx