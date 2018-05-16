'''
This file contains some constants used in the processing
'''
import scipy.constants as const
import numpy as np

solar_irradiance = {'ats': 254.752,
                    'at2': 249.604,
                    'at1': 250.728}

day_night_angle = 101  # solar zenith angles must be > 101 deg VIIRS night-time lights Elvidge et al., 2017

# flare detection limit.  We do not use the upper threshold of 3% as this removes a number of
# significant flares pixels.  E.g. over Iraq.  TODO make a visualisation to show this!
swir_thresh_ats = 0.06  # based on min value in ATS

# set radiance thresholds
swir_thresh_sls = 4 * swir_thresh_ats  # pixel are factor of four smaller, so increase by factor of 4 to make similar
s5_rad_thresh = swir_thresh_sls / (100 * np.pi) * 254.23103333
s6_rad_thresh = swir_thresh_sls / (100 * np.pi) * 80.24474

frp_coeff = {'sls': const.sigma / 8.19919059044e-09,
             'ats': const.sigma / 8.29908189231e-09,
             'at2': const.sigma / 8.2215268253e-09,
             'at1': const.sigma / 8.23565040885e-09}


